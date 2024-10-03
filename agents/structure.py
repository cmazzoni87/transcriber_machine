from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional
import datetime

today = datetime.date.today().strftime("%Y-%m-%d")


class ActionItems(BaseModel):
    description: str = Field(description="Brief description of the action item")
    responsible_party: str = Field(description="Person responsible for the task")
    deadline: Optional[str] = Field(None, description=f"Deadline or follow-up date for the action item, if any, cannot be before {today}")
    additional_notes: Optional[str] = Field(None, description="Any additional notes or details")

    @field_validator('deadline', mode='after')
    def check_deadline(cls, v):
        if v:
            try:
                deadline_date = datetime.datetime.strptime(v, "%Y-%m-%d").date()
                _today = datetime.datetime.today().date()
                if deadline_date < _today:
                    raise ValueError("Deadline cannot be in the past.")
            except ValueError:
                raise ValueError("Deadline must be in YYYY-MM-DD format.")
        return v


class SentimentDetail(BaseModel):
    speaker: str = Field(description="Name or identifier of the speaker")
    sentiment: str = Field(description="Sentiment expressed by the speaker (positive, negative, neutral)")
    remarks: str = Field(description="Specific remarks that contributed to the sentiment")


class SentimentAnalysis(BaseModel):
    overall_sentiment: str = Field(description="Overall sentiment of the conversation (positive, negative, neutral)")
    detailed_sentiment: List[SentimentDetail] = Field(description="Detailed sentiment analysis for each speaker")


class ConversationSummary(BaseModel):
    topic: str = Field(description="Main topic or theme of the conversation")
    summary: str = Field(description="The summary of the discussion")


class PotentialPriority(BaseModel):
    priority_level: str = Field(description="High/Medium/Low")
    description: str = Field(description="Description of the priority")
    related_action_items: List[str] = Field(description="Reference to related action items, if any")
    strategic_importance: str = Field(description="Why this priority is important for achieving the meeting's goals")


class Priorities(BaseModel):
    potential_priorities: List[PotentialPriority] = Field(description="Organized potential priorities to be addressed")


class PotentialDecisions(BaseModel):
    decision: str = Field(description="Name of the decision")
    description: str = Field(description="Description of the decision")
    reasoning: str = Field(description="Reasoning behind the decision")


class KeyDecisions(BaseModel):
    key_decisions: List[PotentialDecisions] = Field(description="Organized key decisions made during the meeting")


class MeetingAnalysis(BaseModel):
    action_items: List[ActionItems] = Field(description="List of action items extracted from the meeting")
    sentiment_analysis: SentimentAnalysis = Field(description="Analysis of sentiment throughout the conversation")
    # conversation_summary: List[ConversationSummary] = Field(description="Summary of the meeting")
    potential_priorities: List[PotentialPriority] = Field(description="Organized potential priorities to be addressed")



