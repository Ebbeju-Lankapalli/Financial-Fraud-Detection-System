"""
Fraud investigation agent.

The QLoRA fraud classifier remains the decision source. Groq is used
only to generate a grounded explanation and investigation guidance.
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


class FraudAgentError(
    RuntimeError
):
    """Raised when the fraud agent cannot complete its workflow."""


class FraudAgent:
    """Coordinate QLoRA classification and LLM explanation."""

    def __init__(
        self,
        *,
        analysis_tool: AnalyzeTransactionTool | None = None,
        groq_client=None,
    ) -> None:
        self.analysis_tool = (
            analysis_tool
            or AnalyzeTransactionTool()
        )

        self.groq_client = groq_client

        if (
            self.groq_client is None
            and settings.groq_api_key
        ):
            self.groq_client = Groq(
                api_key=settings.groq_api_key,
                timeout=settings.groq_timeout_seconds,
            )

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
        """Run fraud classification then generate investigation guidance."""

        analysis = self.analysis_tool.run(
            request.transaction
        )

        if self.groq_client is None:
            explanation = (
                self._fallback_explanation(
                    request,
                    analysis,
                )
            )

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
                agent_model=None,
                llm_used=False,
            )

        try:
            explanation = self._generate_explanation(
                request,
                analysis,
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

        user_content = (
            "Transaction:\n"
            f"- Type: {transaction.type}\n"
            f"- Amount: {transaction.amount:.2f}\n"
            "- Sender balance before: "
            f"{transaction.oldbalanceOrg:.2f}\n"
            "- Sender balance after: "
            f"{transaction.newbalanceOrig:.2f}\n"
            "- Recipient balance before: "
            f"{transaction.oldbalanceDest:.2f}\n"
            "- Recipient balance after: "
            f"{transaction.newbalanceDest:.2f}\n\n"
            "Fraud classifier decision:\n"
            f"- Risk: {analysis.prediction.risk}\n"
            f"- Model: {analysis.prediction.model}\n\n"
        )

        if request.question:
            user_content += (
                "Investigator question:\n"
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
        risk = analysis.prediction.risk

        return (
            f"The fine-tuned fraud classifier marked this "
            f"transaction as {risk} risk. "
            f"It is a {transaction.type} transaction for "
            f"${transaction.amount:,.2f}. "
            "The decision should be treated as a screening signal "
            "and reviewed together with the recorded transaction "
            "balances and account context."
        )

    @staticmethod
    def _default_recommendations(
        analysis: TransactionAnalysisResponse,
    ) -> list[str]:
        """Return deterministic operational recommendations."""

        if analysis.prediction.risk == "HIGH":
            return [
                "Review the transaction and account history.",
                "Verify the sender and recipient activity.",
                "Escalate for manual fraud investigation if warranted.",
            ]

        return [
            "Continue standard transaction monitoring.",
            "Retain the analysis record for audit history.",
        ]
