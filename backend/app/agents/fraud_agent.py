"""
Fraud investigation agent.

The production CatBoost IEEE-CIS classifier is the fraud decision source.
Groq is used only to generate grounded explanation and investigation guidance.
"""

from __future__ import annotations

import logging

from app.agents.prompts import (
    SYSTEM_PROMPT,
)
from app.core.config import settings
from app.schemas.agent import (
    AgentAnalysisRequest,
    AgentAnalysisResponse,
)
from app.schemas.prediction import (
    TransactionAnalysisResponse,
)
from app.tools.analyze_transaction import (
    AnalyzeTransactionTool,
)
from groq import Groq

LOGGER = logging.getLogger(__name__)

_AUTO_GROQ_CLIENT = object()


class FraudAgentError(
    RuntimeError
):
    """Raised when the fraud agent cannot complete its workflow."""


class FraudAgent:
    """Coordinate CatBoost classification and LLM explanation."""

    def __init__(
        self,
        *,
        analysis_tool: AnalyzeTransactionTool | None = None,
        groq_client=_AUTO_GROQ_CLIENT,
    ) -> None:
        self.analysis_tool = (
            analysis_tool
            or AnalyzeTransactionTool()
        )

        if groq_client is _AUTO_GROQ_CLIENT:
            if settings.groq_api_key:
                self.groq_client = Groq(
                    api_key=settings.groq_api_key,
                    timeout=settings.groq_timeout_seconds,
                )
            else:
                self.groq_client = None
        else:
            self.groq_client = groq_client

    @property
    def groq_configured(
        self,
    ) -> bool:
        """Return whether Groq explanation generation is available."""

        return self.groq_client is not None

    def analyze(
        self,
        request: AgentAnalysisRequest,
    ) -> AgentAnalysisResponse:
        """Run CatBoost fraud screening and investigation guidance."""

        analysis = self.analysis_tool.run(
            request.transaction
        )

        if self.groq_client is None:
            return AgentAnalysisResponse(
                analysis_id=analysis.analysis_id,
                created_at=analysis.created_at,
                prediction=analysis.prediction,
                explanation=(
                    self._fallback_explanation(
                        request,
                        analysis,
                    )
                ),
                recommendations=(
                    self._default_recommendations(
                        analysis
                    )
                ),
                agent_model=None,
                llm_used=False,
            )

        try:
            explanation = (
                self._generate_explanation(
                    request,
                    analysis,
                )
            )

        except Exception as exc:
            LOGGER.exception(
                "Groq fraud-agent request failed."
            )

            raise FraudAgentError(
                "Fraud-agent explanation service failed."
            ) from exc

        return AgentAnalysisResponse(
            analysis_id=analysis.analysis_id,
            created_at=analysis.created_at,
            prediction=analysis.prediction,
            explanation=explanation,
            recommendations=(
                self._default_recommendations(
                    analysis
                )
            ),
            agent_model=settings.groq_model,
            llm_used=True,
        )

    def _generate_explanation(
        self,
        request: AgentAnalysisRequest,
        analysis: TransactionAnalysisResponse,
    ) -> str:
        """Generate a grounded explanation using Groq."""

        transaction = request.transaction
        prediction = analysis.prediction

        user_content = (
            "Production fraud screening result:\n"
            f"- Risk: {prediction.risk}\n"
            "- Fraud probability: "
            f"{prediction.fraud_probability:.4f}\n"
            f"- Decision threshold: {prediction.threshold:.2f}\n"
            f"- Model: {prediction.model}\n"
            "- Dataset family: IEEE-CIS Fraud Detection\n\n"
            "Observed transaction fields:\n"
            "- Transaction amount: "
            f"{transaction.transaction_amt:.2f}\n"
            f"- Card1: {transaction.card1}\n"
            f"- Card2: {transaction.card2}\n"
            f"- Card5: {transaction.card5}\n"
            f"- Address feature: {transaction.addr1}\n"
            "- Purchaser email domain: "
            f"{transaction.purchaser_email_domain}\n\n"
            "Important instruction: C*, D*, and M*_enc fields are "
            "anonymized IEEE-CIS features. Do not invent a business "
            "meaning for them. Treat them only as model inputs.\n"
        )

        if request.question:
            user_content += (
                "\nInvestigator question:\n"
                f"{request.question}\n"
            )

        completion = (
            self.groq_client
            .chat
            .completions
            .create(
                model=settings.groq_model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
                reasoning_format="hidden",
                reasoning_effort="low",
                temperature=0.2,
                max_tokens=500,
            )
        )

        content = (
            completion
            .choices[0]
            .message
            .content
        )

        if not content:
            raise FraudAgentError(
                "Groq returned an empty explanation."
            )

        return content.strip()

    @staticmethod
    def _fallback_explanation(
        request: AgentAnalysisRequest,
        analysis: TransactionAnalysisResponse,
    ) -> str:
        """Produce a deterministic explanation without an external LLM."""

        transaction = request.transaction
        prediction = analysis.prediction

        return (
            "The production CatBoost fraud detector marked this "
            f"transaction as {prediction.risk} risk with a "
            f"{prediction.fraud_probability:.2%} fraud probability. "
            f"The decision threshold is {prediction.threshold:.2f}. "
            f"The transaction amount is "
            f"${transaction.transaction_amt:,.2f}. "
            "This is a screening result rather than proof of fraud, "
            "and should be reviewed with account and transaction context."
        )

    @staticmethod
    def _default_recommendations(
        analysis: TransactionAnalysisResponse,
    ) -> list[str]:
        """Return deterministic operational recommendations."""

        if analysis.prediction.risk == "HIGH":
            return [
                "Review the transaction and related account history.",
                "Verify card, address, and purchaser activity.",
                "Escalate for manual fraud investigation if warranted.",
            ]

        return [
            "Continue standard transaction monitoring.",
            "Retain the analysis record for audit history.",
        ]
