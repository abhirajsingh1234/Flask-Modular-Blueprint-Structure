from app.extensions import db
from datetime import datetime

class ConversationHistory(db.Model):
    __tablename__ = 'Conversation_History'
    __table_args__ = {"schema": "dbo"}
    convo_id = db.Column(db.BigInteger, nullable=False)
    msg_id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    human_message = db.Column(db.Text, nullable=False)
    ai_message = db.Column(db.Text, nullable=False)
    user_name = db.Column(db.String(100), nullable=False)
    request_datetime = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    response_datetime = db.Column(db.DateTime, nullable=True)


    