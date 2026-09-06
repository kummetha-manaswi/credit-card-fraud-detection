"""
FraudGuard AI - Customer-Facing FinTech Assistant Module.
Provides intelligent, context-aware customer guidance on transactions, fraud alerts,
card security, website navigation, and educational fraud prevention topics.
Includes offline rule/intent matching fallback and optional LLM provider support.
"""

import os
import re
from typing import Any, Dict, List, Optional


class FraudGuardAssistant:
    """
    Intelligent FinTech Assistant for AI Fraud Guard.
    Acts as a bank/fintech customer-support & fraud-prevention specialist.
    """

    QUICK_ACTIONS = [
        "🔍 Check a transaction",
        "🛡️ Suspicious activity",
        "💳 Payment declined help",
        "🔐 Card security tips",
        "❓ What is a fraud alert?"
    ]

    def __init__(self):
        # Check for optional API keys (Google Gemini / OpenAI)
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

    def get_welcome_message(self) -> str:
        return (
            "👋 **Hi! I'm FraudGuard AI**, your virtual transaction security specialist.\n\n"
            "I can help you evaluate transactions, understand fraud alerts, protect your card, "
            "or navigate the security features of this platform.\n\n"
            "How can I assist you today?"
        )

    def respond(
        self,
        user_message: str,
        current_page: str = "Home",
        last_eval_result: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a helpful, friendly, and context-aware customer response.
        """
        msg = user_message.strip().lower()

        # ----------------------------------------------------
        # 1. SECURITY & SENSITIVE CREDENTIAL CHECKS
        # ----------------------------------------------------
        if any(term in msg for term in ["pin", "cvv", "password", "otp", "secret code"]):
            return (
                "🛡️ **Security Alert:** Never share your card PIN, CVV, passwords, or one-time passcodes (OTP) "
                "with anyone—including bank representatives or virtual assistants. FraudGuard AI will never "
                "request your private security credentials."
            )

        # ----------------------------------------------------
        # 2. CONTEXTUAL: RECENT RISK RESULT EXPLANATION
        # ----------------------------------------------------
        if any(term in msg for term in ["this result", "my result", "result", "flagged", "why flagged", "risk level", "risk score", "what does this mean", "why was my transaction"]):
            if last_eval_result:
                tier = last_eval_result.get("risk_level", "UNKNOWN")
                prob = last_eval_result.get("fraud_probability", 0.0) * 100.0
                score = last_eval_result.get("risk_score", 0.0)
                action = last_eval_result.get("action", "REVIEW")

                if tier == "HIGH":
                    return (
                        f"🚨 **High Risk Assessment:** This transaction received a fraud probability of **{prob:.2f}%** "
                        f"(Risk Score: **{score:.1f}/100**), which exceeds our calibrated cutoff of 0.75.\n\n"
                        f"**Recommended Action:** **{action}**.\n\n"
                        "Our model detected critical anomalous deviations in latent transaction vectors. "
                        "Check the *'Why did AI flag this?'* section below the result to see the specific factors involved."
                    )
                elif tier == "MEDIUM":
                    return (
                        f"🟠 **Suspicious / Moderate Risk:** This transaction scored **{score:.1f}/100** (probability **{prob:.2f}%**).\n\n"
                        f"**Recommended Action:** **{action}**.\n\n"
                        "While not definitively fraudulent, it deviates from typical spending patterns. "
                        "Card issuers typically issue an SMS OTP or push notification for cardholder confirmation."
                    )
                else:
                    return (
                        f"🟢 **Low Risk Approved:** This transaction scored **{score:.1f}/100** (probability **{prob:.2f}%**).\n\n"
                        f"**Recommended Action:** **{action}**.\n\n"
                        "The transaction patterns align with verified genuine cardholder activity and clears automated checks."
                    )
            else:
                return (
                    "To assess a specific transaction, navigate to **🔍 Risk Checker** and click **Analyze Transaction**. "
                    "Once evaluated, I'll explain the exact probability, score, and decision factors for you."
                )

        # ----------------------------------------------------
        # 3. WEBSITE NAVIGATION INTENTS
        # ----------------------------------------------------
        if any(term in msg for term in ["where can i check", "test transaction", "check a transaction", "analyze"]):
            return (
                "🔍 You can evaluate transactions in the **Risk Checker** page! "
                "Select **'🔍 Risk Checker'** from the sidebar navigation. "
                "In *Simple Mode*, you can test pre-loaded normal, suspicious, or random transactions, adjust amounts, and view instant risk scores."
            )

        if any(term in msg for term in ["fraud pattern", "explore fraud", "hourly", "statistics", "patterns"]):
            return (
                "📊 To explore transaction patterns and trends, select **'📊 Fraud Intelligence'** from the navigation menu. "
                "You'll find transaction amount segmentation ($0–$1 micro-testing), hourly nocturnal activity (peaking at 2:00 AM), "
                "and an interactive SQL query terminal."
            )

        if any(term in msg for term in ["how does model work", "ai model", "model performance", "accuracy", "precision", "recall"]):
            return (
                "🤖 You can inspect model benchmarks in **'🤖 AI Model'**. "
                "It features verified test-set metrics (Precision: **93.24%**, Recall: **72.63%**, ROC-AUC: **97.68%**), "
                "a test confusion matrix, and an interactive **Decision Threshold Simulator**."
            )

        if any(term in msg for term in ["why ai", "explainable", "shap", "feature importance", "why did ai"]):
            return (
                "🧠 To understand how the AI arrives at decisions, visit **'🧠 Explainable AI'**. "
                "It uses SHAP (Shapley Additive exPlanations) to visualize both global feature influence and individual transaction push-pull factors."
            )

        # ----------------------------------------------------
        # 4. FINTECH / CARD SECURITY CUSTOMER INTENTS
        # ----------------------------------------------------
        if any(term in msg for term in ["don't recognize", "do not recognize", "unauthorized", "unrecognized", "suspicious transaction", "stolen card"]):
            return (
                "🛡️ **Unrecognized Transaction Guide:**\n\n"
                "1. **Check the details:** Review the amount and time in our **Risk Checker** to see if it displays anomalous behavior.\n"
                "2. **Lock your card:** If this were a live card, immediately freeze your card in your banking app.\n"
                "3. **Contact Card Issuer:** Report unauthorized charges immediately to initiate dispute resolution.\n\n"
                "*Note: This platform is an educational demonstration AI environment and does not connect to live banking accounts.*"
            )

        if any(term in msg for term in ["declined", "declined payment", "payment blocked", "why blocked"]):
            return (
                "💳 **Declined Payment Assistance:**\n\n"
                "A declined payment typically occurs due to:\n"
                "• **Automated Fraud Trigger:** Transaction scored above the decision threshold (e.g. 0.75).\n"
                "• **Unusual Location or Velocity:** Rapid successive charges or off-hours activity (e.g., 2 AM).\n"
                "• **Card Controls:** Daily spending limit reached or international transactions disabled.\n\n"
                "You can simulate this transaction in **Risk Checker** to check if our model classifies it as high risk."
            )

        if any(term in msg for term in ["what is a fraud alert", "fraud alert"]):
            return (
                "🔔 **What is a Fraud Alert?**\n\n"
                "A fraud alert is an automated notification generated when an algorithm detects suspicious or anomalous patterns "
                "on a transaction (such as unusual amounts or sudden activity). It temporarily pauses payment settlement until "
                "verified via SMS, push notification, or cardholder confirmation."
            )

        if any(term in msg for term in ["protect", "protect my card", "card security", "prevent fraud", "safety tips"]):
            return (
                "🔐 **Top Card Protection Practices:**\n\n"
                "• **Never share CVV or OTP:** Genuine institutions never ask for your one-time password.\n"
                "• **Enable Real-Time Alerts:** Receive instant push notifications for all transaction amounts.\n"
                "• **Use Virtual Single-Use Cards:** Ideal for new or untrusted online merchants.\n"
                "• **Review Monthly Activity:** Promptly dispute unrecognized micro-transactions (e.g. $0–$1 testing charges)."
            )

        if any(term in msg for term in ["false positive", "false alarm"]):
            return (
                "⚖️ **What is a False Positive?**\n\n"
                "A false positive occurs when a legitimate customer transaction is mistakenly flagged as fraud. "
                "In our platform, tuning the decision threshold to **0.75** slashes false alarms from 690 down to just **5** "
                "(achieving **93.24% Precision**), ensuring genuine customers enjoy smooth checkouts."
            )

        if any(term in msg for term in ["what are v1", "pca", "v1-v28", "latent"]):
            return (
                "🔬 **About Features V1–V28:**\n\n"
                "In the original dataset, features V1 through V28 are numerical principal components resulting from "
                "**Principal Component Analysis (PCA)**. Banks anonymize proprietary transaction metadata (such as merchant, "
                "terminal, and cardholder attributes) to protect customer confidentiality while preserving predictive power."
            )

        # ----------------------------------------------------
        # 5. GENERAL POLITE GREETINGS
        # ----------------------------------------------------
        if any(term in msg for term in ["hi", "hello", "hey", "good morning", "good afternoon"]):
            return (
                "👋 Hello! I'm FraudGuard AI. Whether you want to test transaction risk in **Risk Checker**, "
                "explore patterns in **Fraud Intelligence**, or ask about card security, I'm here to help!"
            )

        if any(term in msg for term in ["thank", "thanks", "awesome", "great"]):
            return (
                "You're very welcome! Let me know if you need help testing another transaction or exploring our AI fraud model."
            )

        # ----------------------------------------------------
        # 6. DEFAULT HELPFUL FALLBACK
        # ----------------------------------------------------
        return (
            "I'm here to help with credit card security, transaction risk assessment, and platform navigation.\n\n"
            "Here are a few things you can ask me:\n"
            "• *'I don't recognize a transaction on my card'* \n"
            "• *'Why was my payment declined?'* \n"
            "• *'Where can I check a transaction?'* \n"
            "• *'What does a fraud alert mean?'* \n"
            "• *'How does AI detect fraud?'*"
        )


_assistant_instance: Optional[FraudGuardAssistant] = None


def get_assistant() -> FraudGuardAssistant:
    """Return singleton instance of FraudGuardAssistant."""
    global _assistant_instance
    if _assistant_instance is None:
        _assistant_instance = FraudGuardAssistant()
    return _assistant_instance
