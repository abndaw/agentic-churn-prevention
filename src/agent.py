"""
Retention agent: ML prediction + SHAP explanation + LLM email generation.

Uses POLICY LAYER (validation-tuned thresholds).
Raw data is loaded to verify customer eligibility for offers.

Pipeline:
  predict churn -> threshold decision -> SHAP explanation ->
  segment detection -> offer ranking (with raw data checks) -> email generation
"""

import os
import json
import shap
import pickle
import pandas as pd
from pathlib import Path
from google import genai
from dotenv import load_dotenv
from preprocess import preprocess

load_dotenv()

# Segments
SEGMENTS = {
    'fiber_new_senior': {'name': 'Ultra-High Risk', 'tone': 'supportive_senior'},
    'fiber_new': {'name': 'High Risk', 'tone': 'energetic_tech'},
    'new_senior': {'name': 'Elevated Risk', 'tone': 'supportive_senior'},
    'new': {'name': 'Moderate Risk', 'tone': 'friendly'},
    'standard': {'name': 'Standard', 'tone': 'friendly'},
}


def identify_segment(row):
    """Identify customer segment based on risk factors"""
    is_fiber = row.get('InternetService_Fiber optic', 0) == 1
    is_new = row.get('tenure', 999) < 12
    is_senior = row.get('SeniorCitizen', 0) == 1

    if is_fiber and is_new and is_senior:
        return SEGMENTS['fiber_new_senior']
    if is_fiber and is_new:
        return SEGMENTS['fiber_new']
    if is_new and is_senior:
        return SEGMENTS['new_senior']
    if is_new:
        return SEGMENTS['new']
    return SEGMENTS['standard']


# OFFER FUNCTIONS
def _contract_upgrade(monthly, senior=False):
    """Contract: MTM → 1 year commitment"""
    return {
        'name': '1-Year Rate Lock',
        'benefit': 'Freeze your monthly price and avoid future increases',
        'discount': '$5/month off' if not senior else f'10% off (~${monthly * 0.10:.2f})'
    }


def _contract_extension(contract, senior=False):
    """Contract: Extend existing 1 year or 2 year contract"""
    return {
        'name': 'Early Renewal Bonus',
        'benefit': 'Extend your contract now and lock in your rate',
        'discount': '$50 account credit' if not senior else '$75 account credit'
    }


def _loyalty_bonus(tenure):
    """Tenure: New customers (< 12 months)"""
    return {
        'name': 'New Customer Loyalty Bonus',
        'benefit': 'Lock in savings as a thank you for joining us',
        'discount': '$10/month off for 12 months'
    }


def _anniversary_rewards(tenure):
    """Tenure: Established customers (≥ 12 months)"""
    years = tenure // 12
    credit = years * 20
    return {
        'name': f'{years} Year Loyalty Reward',
        'benefit': 'Thank you for your continued business',
        'discount': f'${credit} account credit'
    }


def _rate_reduction(monthly):
    """Pricing: High monthly charges"""
    amount = min(monthly * 0.10, 15)
    return {
        'name': 'Valued Customer Discount',
        'benefit': 'Lower your rate as a thank you for your business',
        'discount': f'${amount:.0f}/month off'
    }


def _fiber_upgrade():
    """Internet: DSL -> Fiber upgrade"""
    return {
        'name': 'Free Fiber Upgrade',
        'benefit': '10x faster speeds with fiber technology',
        'discount': 'Free installation ($99 value)'
    }


def _internet_signup():
    """Internet: No service -> Sign up"""
    return {
        'name': 'Internet Starter Pack',
        'benefit': 'Get connected with high speed internet',
        'discount': 'First 3 months at 50% off'
    }


def _speed_upgrade():
    """Internet: Existing -> Faster tier"""
    return {
        'name': 'Free Speed Boost',
        'benefit': 'Upgrade to the next speed tier',
        'discount': 'Free for 3 months'
    }


def _autopay_discount():
    """Payment: Manual -> Automatic"""
    return {
        'name': 'Auto-Pay Discount',
        'benefit': 'Never miss a payment and save every month automatically',
        'discount': '$5/month ongoing'
    }


def _security_bundle(senior=False):
    """Service: Add online security"""
    return {
        'name': 'Security & Backup Bundle',
        'benefit': 'Protect your devices and automatically back up important files',
        'discount': 'First 3 months free ($30 value)' if not senior else '6 months free ($60 value)'
    }


def _online_backup():
    """Service: Add backup"""
    return {
        'name': 'Cloud Backup Service',
        'benefit': 'Automatically backup your important files to the cloud',
        'discount': 'First 2 months free ($20 value)'
    }


def _tech_support():
    """Service: Add tech support"""
    return {
        'name': 'Priority Tech Support',
        'benefit': '24/7 expert help when you need it most',
        'discount': '$3/month for first year'
    }


def _device_protection():
    """Service: Add device protection"""
    return {
        'name': 'Device Protection Plan',
        'benefit': 'Coverage for accidental damage and technical issues',
        'discount': 'First 3 months free ($25 value)'
    }


def _streaming_bundle():
    """Service: Add streaming"""
    return {
        'name': 'Premium Streaming Bundle',
        'benefit': 'Unlimited 4K streaming for TV shows and movies',
        'discount': 'Add both for $15/month (normally $25)'
    }


def _paperless_billing():
    """Billing: Paper -> Paperless"""
    return {
        'name': 'Paperless Billing Bonus',
        'benefit': 'Go green and get rewarded',
        'discount': '$5 account credit'
    }


def _referral_program():
    """Universal: Always available"""
    return {
        'name': 'Refer & Earn',
        'benefit': 'Get rewarded for sharing',
        'discount': '$50 for each friend who signs up'
    }


# RETENTION AGENT

class RetentionAgent:

    def __init__(self, model_path, api_key=None, raw_data_path=None,
                 threshold_path="../outputs/threshold_results.json"):
        """
        Initialize RetentionAgent with model and validation-tuned thresholds.
        
        Args:
            model_path: Path to trained model file
            api_key: Google Gemini API key for LLM (from environment)
            raw_data_path: Path to raw data CSV for eligibility checking
            threshold_path: Path to threshold_results.json from validation optimization
        """
        
        # Loading the model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Try to load metadata, fallback to preprocess if not found
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        try:
            with open(metadata_path) as f:
                self.metadata = json.load(f)
            self.feature_names = self.metadata['feature_names']
        except FileNotFoundError:
            print(f"Metadata not found at {metadata_path}")
            print("Loading feature names from preprocess")
            _, _, _, _, _, _, self.feature_names = preprocess(scale_numeric=False)
            print(f"Loaded {len(self.feature_names)} features")

        # Loading raw data for eligibility checking
        if raw_data_path is None:
            # Default path
            project_root = Path(__file__).parent.parent
            raw_data_path = project_root / "data" / "telco_churn.csv"
        
        try:
            self.raw_data = pd.read_csv(raw_data_path)
            # Seting customerID as index
            if 'customerID' in self.raw_data.columns:
                self.raw_data.set_index('customerID', inplace=True)
            print(f"Loaded raw data: {len(self.raw_data)} customers")
        except FileNotFoundError:
            print(f"Raw data not found at {raw_data_path}")
            print("Eligibility checks will use preprocessed features only")
            self.raw_data = None

        # Loading validation-tuned thresholds
        try:
            with open(threshold_path) as f:
                results = json.load(f)
                policy = results['policy']
            
            self.T_HIGH = policy['threshold_high']
            self.T_MED = policy['threshold_medium']
            
            print(f"Loaded thresholds from validation optimization:")
            print(f"High:   {self.T_HIGH:.2f}")
            print(f"Medium: {self.T_MED:.2f}")
        except FileNotFoundError:
            print(f"Threshold file not found at {threshold_path}")
            print("Using thresholds manually (0.90, 0.30)")
            self.T_HIGH = 0.90
            self.T_MED = 0.30

        # Initialize LLM client
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY")

        self.genai = genai.Client(api_key=api_key)
        self.explainer = shap.TreeExplainer(self.model)

    def predict_churn(self, X):
        """Predict churn probability for a customer"""
        return float(self.model.predict_proba(X)[0, 1])

    def decide_action(self, prob):
        """Decide whether to contact customer based on churn probability"""
        if prob >= self.T_HIGH:
            return {"risk": "High", "contact": True}
        if prob >= self.T_MED:
            return {"risk": "Medium", "contact": True}
        return {"risk": "Low", "contact": False}

    def explain(self, X, top_k=5):
        """Get top risk factors using SHAP values"""
        sv = self.explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]

        impacts = []
        for i, name in enumerate(self.feature_names):
            impacts.append({
                "feature": name,
                "shap": float(sv[0, i])
            })

        impacts.sort(key=lambda x: abs(x["shap"]), reverse=True)
        return impacts[:top_k]

    def get_customer_raw_data(self, customer_id):
        """Get raw data for a customer by ID"""
        if self.raw_data is None:
            return None
        
        try:
            return self.raw_data.loc[customer_id]
        except KeyError:
            print(f"Customer {customer_id} not found in raw data")
            return None

    def build_offers(self, X, segment, risk_factors, customer_id=None):
        """
        Build personalized offers: SHAP prioritized + eligibility checked.
        
        Uses SHAP values to prioritize offers, then checks eligibility against raw customer data. Returns top 3 offers.
        """
        
        row = X.iloc[0]
        
        # Getting raw customer data for eligibility checks
        raw_row = None
        if customer_id is not None:
            raw_row = self.get_customer_raw_data(customer_id)
        
        # Extracting positive SHAP risks (features increasing churn)
        top_risks = [rf for rf in risk_factors if rf['shap'] > 0][:5]
        
        # Base priorities from global SHAP importance
        BASE_PRIORITIES = {
            'contract': 100,      # Rank #1
            'tenure': 95,         # Rank #2
            'internet': 85,       # Rank #3
            'payment': 80,        # Rank #4
            'pricing': 75,        # Rank #5
            'security': 60,       # Rank #7
            'streaming': 50,      # Ranks #8-10
            'billing': 45,        # Rank #9
            'support': 40,        # Rank #11
            'backup': 35,         # Rank #20
            'device': 30,         # Lower priority
        }
        
        # Creating priority scores from SHAP (individual customer risks)
        offer_priorities = {}
        
        for risk in top_risks:
            feature = risk['feature'].lower()
            shap_value = abs(risk['shap'])
            
            if 'contract' in feature:
                offer_priorities['contract'] = offer_priorities.get('contract', 0) + shap_value
            if 'tenure' in feature:
                offer_priorities['tenure'] = offer_priorities.get('tenure', 0) + shap_value
            if 'internet' in feature:
                offer_priorities['internet'] = offer_priorities.get('internet', 0) + shap_value
            if 'payment' in feature or 'check' in feature:
                offer_priorities['autopay'] = offer_priorities.get('autopay', 0) + shap_value
            if 'monthly' in feature or 'charges' in feature:
                offer_priorities['pricing'] = offer_priorities.get('pricing', 0) + shap_value
            if 'security' in feature:
                offer_priorities['security'] = offer_priorities.get('security', 0) + shap_value
            if 'backup' in feature:
                offer_priorities['backup'] = offer_priorities.get('backup', 0) + shap_value
            if 'support' in feature:
                offer_priorities['tech_support'] = offer_priorities.get('tech_support', 0) + shap_value
            if 'streaming' in feature:
                offer_priorities['streaming'] = offer_priorities.get('streaming', 0) + shap_value
            if 'paperless' in feature or 'billing' in feature:
                offer_priorities['paperless'] = offer_priorities.get('paperless', 0) + shap_value
            if 'device' in feature or 'protection' in feature:
                offer_priorities['device'] = offer_priorities.get('device', 0) + shap_value
        
        print(f"\nOffer priorities (SHAP): {offer_priorities}")
        
        # Get customer info
        if raw_row is not None:
            tenure = int(raw_row.get('tenure', row.get('tenure', 0)))
            monthly = float(raw_row.get('MonthlyCharges', row.get('MonthlyCharges', 70)))
            is_senior = int(raw_row.get('SeniorCitizen', row.get('SeniorCitizen', 0))) == 1
            contract = str(raw_row.get('Contract', '')).strip()
            internet = str(raw_row.get('InternetService', '')).strip()
            security = str(raw_row.get('OnlineSecurity', '')).strip()
            backup = str(raw_row.get('OnlineBackup', '')).strip()
            support = str(raw_row.get('TechSupport', '')).strip()
            device_protection = str(raw_row.get('DeviceProtection', '')).strip()
            payment = str(raw_row.get('PaymentMethod', '')).strip()
            streaming_tv = str(raw_row.get('StreamingTV', '')).strip()
            streaming_movies = str(raw_row.get('StreamingMovies', '')).strip()
            paperless = str(raw_row.get('PaperlessBilling', '')).strip()
        else:
            # Fallback to preprocessed features
            tenure = int(row.get('tenure', 0))
            monthly = float(row.get('MonthlyCharges', 70))
            is_senior = row.get('SeniorCitizen', 0) == 1
            contract = "Month-to-month" if row.get('Contract_Month-to-month', 0) == 1 else "One year"
            internet = "Fiber optic" if row.get('InternetService_Fiber optic', 0) == 1 else "DSL"
            security = "Yes" if row.get('OnlineSecurity_Yes', 0) == 1 else "No"
            backup = "Yes" if row.get('OnlineBackup_Yes', 0) == 1 else "No"
            support = "Yes" if row.get('TechSupport_Yes', 0) == 1 else "No"
            device_protection = "Yes" if row.get('DeviceProtection_Yes', 0) == 1 else "No"
            payment = "Electronic check" if row.get('PaymentMethod_Electronic check', 0) == 1 else "Unknown"
            streaming_tv = "Yes" if row.get('StreamingTV_Yes', 0) == 1 else "No"
            streaming_movies = "Yes" if row.get('StreamingMovies_Yes', 0) == 1 else "No"
            paperless = "Yes" if row.get('PaperlessBilling_Yes', 0) == 1 else "No"
        
        print(f"\nCustomer profile:")
        print(f"  Tenure: {tenure} months")
        print(f"  Contract: {contract}")
        print(f"  Internet: {internet}")
        print(f"  Monthly: ${monthly:.2f}")
        print(f"  Senior: {is_senior}")
        
        # Building eligible offers with priorities
        eligible_offers = []
        
        has_internet = internet in ["DSL", "Fiber optic"]
        
        # 1. Contract Upgrade (MTM -> 1 year)
        if contract == "Month-to-month":
            priority = BASE_PRIORITIES['contract'] + offer_priorities.get('contract', 0)
            eligible_offers.append({
                'priority': priority,
                'offer': _contract_upgrade(monthly, senior=is_senior)
            })
            print(f"  ✓ Contract upgrade: priority={priority:.2f}")
        
        # 2. Contract Extension (non-MTM)
        elif contract in ["One year", "Two year"]:
            priority = BASE_PRIORITIES['contract'] + offer_priorities.get('contract', 0)
            eligible_offers.append({
                'priority': priority,
                'offer': _contract_extension(contract, senior=is_senior)
            })
            print(f"  ✓ Contract extension: priority={priority:.2f}")
        
        # 3. Loyalty Bonus (new customers)
        if tenure < 12:
            priority = BASE_PRIORITIES['tenure'] + offer_priorities.get('tenure', 0)
            eligible_offers.append({
                'priority': priority,
                'offer': _loyalty_bonus(tenure)
            })
            print(f"  ✓ Loyalty bonus: priority={priority:.2f}")
        
        # 4. Anniversary Rewards (established customers)
        elif tenure >= 12:
            priority = BASE_PRIORITIES['tenure'] + offer_priorities.get('tenure', 0)
            eligible_offers.append({
                'priority': priority,
                'offer': _anniversary_rewards(tenure)
            })
            print(f"  ✓ Anniversary rewards: priority={priority:.2f}")
        
        # 5. Fiber Upgrade (DSL -> Fiber) 
        if internet == "DSL":
            # Get SHAP for fiber feature
            fiber_shap = next(
                (rf['shap'] for rf in risk_factors if 'Fiber optic' in rf['feature']),
                0
            )
            
            # Only offer if SHAP is positive (not having fiber is a risk)
            if fiber_shap > 0:
                priority = BASE_PRIORITIES['internet'] + fiber_shap
                eligible_offers.append({
                    'priority': priority,
                    'offer': _fiber_upgrade()
                })
                print(f"  ✓ Fiber upgrade: priority={priority:.2f} (SHAP={fiber_shap:.3f})")
            else:
                print(f"  ✗ Fiber upgrade: SHAP={fiber_shap:.3f} indicates DSL is protective")

        # 6. Internet Signup (No internet -> Sign up)
        elif internet == "No":
            # Get SHAP for no internet feature
            no_internet_shap = next(
                (rf['shap'] for rf in risk_factors if 'InternetService_No' in rf['feature']),
                0
            )
            
            # Only offer if SHAP is positive (no internet is a risk)
            if no_internet_shap > 0:
                priority = BASE_PRIORITIES['internet'] + no_internet_shap
                eligible_offers.append({
                    'priority': priority,
                    'offer': _internet_signup()
                })
                print(f"  ✓ Internet signup: priority={priority:.2f} (SHAP={no_internet_shap:.3f})")
            else:
                print(f"  ✗ Internet signup: SHAP={no_internet_shap:.3f} indicates no service is OK")
        
        # 7. Speed Upgrade (existing internet)
        elif has_internet:
            priority = BASE_PRIORITIES['internet'] + offer_priorities.get('internet', 0) - 20  # Lower than fiber/signup
            eligible_offers.append({
                'priority': priority,
                'offer': _speed_upgrade()
            })
            print(f"  ✓ Speed upgrade: priority={priority:.2f}")
        
        # 8. Rate Reduction (high monthly charges)
        if monthly > 70:
            priority = BASE_PRIORITIES['pricing'] + offer_priorities.get('pricing', 0)
            eligible_offers.append({
                'priority': priority,
                'offer': _rate_reduction(monthly)
            })
            print(f"  ✓ Rate reduction: priority={priority:.2f}")
        
        # 9. Auto-Pay (manual payment)
        if payment in ["Electronic check", "Mailed check"]:
            priority = BASE_PRIORITIES['payment'] + offer_priorities.get('autopay', 0)
            eligible_offers.append({
                'priority': priority,
                'offer': _autopay_discount()
            })
            print(f"  ✓ Autopay: priority={priority:.2f}")
        
        # 10. Security Bundle
        if has_internet and security == "No":
            priority = BASE_PRIORITIES['security'] + offer_priorities.get('security', 0)
            eligible_offers.append({
                'priority': priority,
                'offer': _security_bundle(senior=is_senior)
            })
            print(f"  ✓ Security: priority={priority:.2f}")
        
        # 11. Online Backup
        if has_internet and backup == "No":
            priority = BASE_PRIORITIES['backup'] + offer_priorities.get('backup', 0)
            eligible_offers.append({
                'priority': priority,
                'offer': _online_backup()
            })
            print(f"  ✓ Backup: priority={priority:.2f}")
        
        # 12. Tech Support
        if has_internet and support == "No":
            priority = BASE_PRIORITIES['support'] + offer_priorities.get('tech_support', 0)
            eligible_offers.append({
                'priority': priority,
                'offer': _tech_support()
            })
            print(f"  ✓ Tech support: priority={priority:.2f}")
        
        # 13. Device Protection
        if has_internet and device_protection == "No":
            priority = BASE_PRIORITIES['device'] + offer_priorities.get('device', 0)
            eligible_offers.append({
                'priority': priority,
                'offer': _device_protection()
            })
            print(f"  ✓ Device protection: priority={priority:.2f}")
        
        # 14. Streaming Bundle
        if internet == "Fiber optic" and streaming_tv == "No" and streaming_movies == "No":
            priority = BASE_PRIORITIES['streaming'] + offer_priorities.get('streaming', 0)
            eligible_offers.append({
                'priority': priority,
                'offer': _streaming_bundle()
            })
            print(f"  ✓ Streaming: priority={priority:.2f}")
        
        # 15. Paperless Billing
        if paperless == "No":
            priority = BASE_PRIORITIES['billing'] + offer_priorities.get('paperless', 0)
            eligible_offers.append({
                'priority': priority,
                'offer': _paperless_billing()
            })
            print(f"  ✓ Paperless: priority={priority:.2f}")
        
        # Sort by priority and take top 3
        eligible_offers.sort(key=lambda x: x['priority'], reverse=True)
        offers = [item['offer'] for item in eligible_offers[:3]]
        
        # Universal fallback if somehow no offers
        if len(offers) == 0:
            print("\n  No eligible offers! Using universal fallback...")
            offers = [_referral_program()]
        
        print(f"\nFinal offers: {[o['name'] for o in offers]}\n")
        
        return offers[:3]

    def generate_email(self, X, risk_factors, segment, offers, use_llm=True):
        """Generate personalized retention email"""
        
        row = X.iloc[0]
        tenure = int(row.get("tenure", 0))

        if use_llm and len(offers) > 0:
            prompt = f"""
Write a natural, friendly retention email.

Rules:
- Human tone, conversational
- No mention of AI, ML, churn, or risk
- Must include greeting, gratitude, offers, closing
- Each offer must include benefit + discount
- Max 180 words
- Sound genuine and helpful, not salesy

Customer:
- {tenure} months with us
- Tone: {segment['tone']}

Offers:
{chr(10).join([f"- {o['name']}: {o['benefit']} ({o['discount']})" for o in offers])}

Format:
Subject: ...
Body: ...
"""

            try:
                response = self.genai.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=prompt
                )
                text = response.text

                if "Subject:" in text and "Body:" in text:
                    subject = text.split("Subject:")[1].split("Body:")[0].strip()
                    body = text.split("Body:")[1].strip()
                    
                    subject = subject.replace("**", "").replace("*", "").strip()
                    body = body.replace("**", "").replace("*", "").strip()
                    
                    return {"subject": subject, "body": body}
                
            except Exception as e:
                print(f"LLM generation failed: {e}, using fallback")

        # Fallback template
        body = (
            f"Hi there,\n\n"
            f"Thanks for being with us for {tenure} months. "
            f"We wanted to share a few ways to improve your experience and save money.\n\n"
        )

        for o in offers:
            body += f"• {o['name']}: {o['benefit']} - {o.get('discount', '')}\n"

        body += (
            "\nIf you'd like to learn more, just reply to this email or give us a call.\n\n"
            "Best regards,\n"
            "Customer Success Team"
        )

        return {
            "subject": "Special offers just for you",
            "body": body
        }

    def analyze(self, customer, customer_id=None):
        """
        Complete analysis pipeline for a customer
        
        Args:
            customer: DataFrame with preprocessed features
            customer_id: Optional customer ID for raw data lookup
        """
        
        if isinstance(customer, dict):
            customer = pd.DataFrame([customer])

        X = customer[self.feature_names]
        
        # Getting customer ID if not provided
        if customer_id is None and hasattr(customer.index, 'name'):
            customer_id = customer.index[0]

        prob = self.predict_churn(X)
        action = self.decide_action(prob)

        if not action["contact"]:
            return {
                "churn_probability": prob,
                "action": action,
                "contacted": False
            }

        risk = self.explain(X)
        segment = identify_segment(X.iloc[0])
        offers = self.build_offers(X, segment, risk, customer_id=customer_id)

        email = self.generate_email(X, risk, segment, offers, use_llm=True)

        return {
            "churn_probability": prob,
            "action": action,
            "segment": segment,
            "risk_factors": risk,
            "offers": offers,
            "email": email,
            "contacted": True
        }


# DEMO

if __name__ == "__main__":
    agent = RetentionAgent(
        "../models/best_model.pkl",
        threshold_path="../outputs/threshold_results.json"
    )

    X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocess()

    # Find a churner from VALIDATION set
    churner_indices = y_val[y_val == 1].index
    if len(churner_indices) > 0:
        customer_id = churner_indices[0]
    else:
        customer_id = y_val.index[0]
    
    customer = X_val.loc[[customer_id]]

    print("RETENTION AGENT DEMO")
    
    result = agent.analyze(customer, customer_id=customer_id)

    print(f"\nCustomer ID: {customer_id}")
    print(f"Churn Probability: {result['churn_probability']:.2%}")
    print(f"Risk Level: {result['action']['risk']}")

    if result["contacted"]:
        print(f"\nSegment: {result['segment']['name']}")
        
        print(f"\nTop Risk Factors:")
        for rf in result["risk_factors"]:
            print(f"  • {rf['feature']}: {rf['shap']:.4f}")
        
        print(f"\nOffers ({len(result['offers'])}):")
        for i, o in enumerate(result["offers"], 1):
            print(f"  {i}. {o['name']}")
            print(f"     {o['benefit']}")
            print(f"     {o['discount']}")
        
        print("EMAIL")
        print(f"\nSubject: {result['email']['subject']}")
        print(f"\n{result['email']['body']}")

    with open("../outputs/sample_retention_email.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\nSaved output")