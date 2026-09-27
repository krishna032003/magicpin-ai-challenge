"""
magicpin AI Challenge — Conversation Handlers
==============================================
Multi-turn conversation management for Vera with robust intent classification,
auto-reply detection, hostile exit handling, bounded state tracking, terminal state
enforcement, and zero-hallucination grounding.
"""

from __future__ import annotations
import re
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone


AUTO_REPLY_PATTERNS = [
    r"thank you for contacting us",
    r"our team will respond shortly",
    r"we will get back to you",
    r"hamari team tak pahuncha",
    r"jaankari ke liye.*shukriya",
    r"automated assistant",
    r"auto-reply",
    r"currently away",
    r"out of office",
    r"please leave a message",
    r"thanks for reaching out",
    r"hum aapse jald hi sampark karenge",
    r"shukriya.*aapka sandesh mil gaya",
    r"busy right now.*reply soon",
]

HOSTILE_PATTERNS = [
    r"\bstop\b",
    r"\bunsubscribe\b",
    r"\bspam\b",
    r"\buseless\b",
    r"stop messaging",
    r"don'?t message",
    r"not interested",
    r"leave me alone",
    r"shut up",
    r"fraud",
    r"block",
    r"bakwas",
    r"band karo",
    r"hatao",
]

COMMITMENT_PATTERNS = [
    r"\bok\b.*let'?s do it",
    r"\blet'?s do it\b",
    r"\bproceed\b",
    r"\bgo ahead\b",
    r"\byes\b",
    r"\bconfirm\b",
    r"\bconfirmed\b",
    r"\bwhats next\b",
    r"\bwhat'?s next\b",
    r"\bdone\b",
    r"\bready\b",
    r"\bsure\b",
    r"\byeah\b",
    r"\bstart\b",
    r"\bstart karo\b",
    r"\bhaan\b",
    r"\btheek hai\b",
    r"\bchalega\b",
    r"\bkardo\b",
    r"\bkar do\b",
    r"\bapply\b",
    r"\bsend me\b",
    r"\bpull it\b",
    r"\bdraft it\b",
    r"\bsign me up\b",
]

WAIT_PATTERNS = [
    r"\bbusy\b",
    r"call.*later",
    r"msg.*later",
    r"message.*later",
    r"talk.*later",
    r"baad mein",
    r"tomorrow",
    r"kal baat",
    r"give me.*time",
    r"next week",
    r"meeting mein hoon",
    r"in a meeting",
]

MAX_TURNS_PER_CONVERSATION = 20
MAX_STORED_CONVERSATIONS = 500


class ConversationManager:
    """Maintains active conversation turns and strictly bounded, terminally enforced state machines."""

    def __init__(self):
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        self.conv_state: Dict[str, Dict[str, Any]] = {}

    def _prune_if_needed(self):
        """Enforces upper bound on stored conversation states to prevent unbounded memory growth."""
        if len(self.conversations) >= MAX_STORED_CONVERSATIONS:
            # Drop the oldest 20% of conversations
            keys_to_remove = list(self.conversations.keys())[:MAX_STORED_CONVERSATIONS // 5]
            for k in keys_to_remove:
                self.conversations.pop(k, None)
                self.conv_state.pop(k, None)

    def get_history(self, conv_id: str) -> List[Dict[str, Any]]:
        return self.conversations.get(conv_id, [])

    def record_turn(self, conv_id: str, from_role: str, message: str, turn_number: int):
        self._prune_if_needed()
        if conv_id not in self.conversations:
            self.conversations[conv_id] = []
        self.conversations[conv_id].append({
            "turn_number": turn_number,
            "from_role": from_role,
            "message": message,
            "ts": datetime.now(timezone.utc).isoformat()
        })
        if len(self.conversations[conv_id]) > MAX_TURNS_PER_CONVERSATION:
            self.conversations[conv_id] = self.conversations[conv_id][-MAX_TURNS_PER_CONVERSATION:]

    def set_state(self, conv_id: str, stage: str, metadata: Optional[Dict[str, Any]] = None):
        self._prune_if_needed()
        self.conv_state[conv_id] = {
            "stage": stage,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **(metadata or {})
        }

    def get_state(self, conv_id: str) -> Dict[str, Any]:
        return self.conv_state.get(conv_id, {"stage": "initial"})

    def is_auto_reply(self, conv_id: str, message: str) -> bool:
        history = self.get_history(conv_id)
        msg_clean = message.strip().lower()

        for pat in AUTO_REPLY_PATTERNS:
            if re.search(pat, msg_clean):
                return True

        merchant_msgs = [t["message"].strip().lower() for t in history if t.get("from_role") == "merchant"]
        if merchant_msgs.count(msg_clean) >= 2:
            return True

        return False

    def is_hostile(self, message: str) -> bool:
        msg_clean = message.strip().lower()
        for pat in HOSTILE_PATTERNS:
            if re.search(pat, msg_clean):
                return True
        return False

    def is_commitment(self, message: str) -> bool:
        msg_clean = message.strip().lower()
        for pat in COMMITMENT_PATTERNS:
            if re.search(pat, msg_clean):
                return True
        return False

    def is_wait_request(self, message: str) -> bool:
        msg_clean = message.strip().lower()
        for pat in WAIT_PATTERNS:
            if re.search(pat, msg_clean):
                return True
        return False

    def clear(self):
        self.conversations.clear()
        self.conv_state.clear()


conv_mgr = ConversationManager()


def respond(
    conversation_id: str,
    merchant_id: Optional[str],
    customer_id: Optional[str],
    from_role: str,
    message: str,
    turn_number: int,
    merchant: Optional[Dict[str, Any]] = None,
    category: Optional[Dict[str, Any]] = None,
    customer: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Produce the next action in a multi-turn conversation.
    Strictly enforces terminal states (ended, opted_out).
    Grounds all outputs in verified merchant, category, and customer context.
    """
    # 0. TERMINAL STATE ENFORCEMENT: prevent ended or opted-out conversations from continuing
    current_state = conv_mgr.get_state(conversation_id)
    current_stage = current_state.get("stage")
    if current_stage == "opted_out":
        return {
            "action": "end",
            "body": "This conversation is unsubscribed. No further messages will be sent.",
            "cta": "none",
            "rationale": "Conversation previously reached terminal opted-out state; rejecting further engagement"
        }
    if current_stage in ["ended", "auto_reply_ended"]:
        return {
            "action": "end",
            "body": "",
            "cta": "none",
            "rationale": "Conversation previously terminated; enforcing terminal end state"
        }

    conv_mgr.record_turn(conversation_id, from_role, message, turn_number)

    merchant_name = ""
    owner_name = ""
    locality = ""
    active_offer = ""
    if merchant:
        ident = merchant.get("identity", {})
        merchant_name = ident.get("name", "")
        owner_name = ident.get("owner_first_name", "")
        locality = ident.get("locality", "")
        offers = merchant.get("offers", [])
        for off in offers:
            if off.get("status") == "active":
                active_offer = off.get("title", "")
                break

    salutation = owner_name or merchant_name or "there"

    # Customer role messages
    if from_role == "customer":
        c_name = customer.get("identity", {}).get("name", "there") if customer else "there"
        if conv_mgr.is_hostile(message):
            conv_mgr.set_state(conversation_id, "opted_out")
            res = {
                "action": "end",
                "body": "Understood. We have removed you from future updates. Have a great day!",
                "cta": "none",
                "rationale": "Customer requested opt-out; permanently closed conversation"
            }
            conv_mgr.record_turn(conversation_id, "vera", res["body"], turn_number + 1)
            return res

        conv_mgr.set_state(conversation_id, "customer_inquiry")
        reply_body = f"Hi {c_name}, thank you for reaching out to {merchant_name or 'us'}. What would you like to know about your booking?"
        res = {
            "action": "send",
            "body": reply_body,
            "cta": "open_ended",
            "rationale": "Courteous customer acknowledgment grounded in merchant context"
        }
        conv_mgr.record_turn(conversation_id, "vera", res["body"], turn_number + 1)
        return res

    # 1. AUTO-REPLY DETECTION
    if conv_mgr.is_auto_reply(conversation_id, message):
        conv_mgr.set_state(conversation_id, "auto_reply_ended")
        return {
            "action": "end",
            "body": "",
            "cta": "none",
            "rationale": "Auto-reply detected from merchant's WhatsApp Business; ending conversation gracefully to avoid burning turns"
        }

    # 2. HOSTILE / OPT-OUT DETECTION
    if conv_mgr.is_hostile(message):
        conv_mgr.set_state(conversation_id, "opted_out")
        res = {
            "action": "end",
            "body": "Understood. Sorry for the disturbance — we won't message you about this again. Have a great day!",
            "cta": "none",
            "rationale": "Merchant signaled opt-out or hostility; permanently marked opted_out"
        }
        conv_mgr.record_turn(conversation_id, "vera", res["body"], turn_number + 1)
        return res

    # 3. WAIT / DEFERRAL DETECTION
    if conv_mgr.is_wait_request(message):
        conv_mgr.set_state(conversation_id, "paused")
        return {
            "action": "wait",
            "wait_seconds": 1800,
            "cta": "none",
            "rationale": "Merchant requested time / is currently busy; backing off 30 minutes"
        }

    # 4. INTENT COMMITMENT (Pattern D: Action-oriented, zero qualifying questions)
    if conv_mgr.is_commitment(message):
        conv_mgr.set_state(conversation_id, "committed")
        store_str = f" for {merchant_name}" if merchant_name else ""
        offer_str = f" featuring '{active_offer}'" if active_offer else ""
        action_body = (
            f"Thanks {salutation}, I have noted your confirmation{store_str}{offer_str}. "
            f"What would you like me to prepare next?"
        )
        res = {
            "action": "send",
            "body": action_body,
            "cta": "open_ended",
            "rationale": "Acknowledges the merchant's confirmation and asks which preparation they want next."
        }
        conv_mgr.record_turn(conversation_id, "vera", res["body"], turn_number + 1)
        return res

    # 5. GENERAL INQUIRY / ENGAGEMENT (Directly grounded in context)
    msg_lower = message.lower()
    if "?" in message or any(q in msg_lower for q in ["how", "what", "where", "kya", "price", "kitna"]):
        conv_mgr.set_state(conversation_id, "answered_inquiry")
        store_info = f" for {merchant_name}" if merchant_name else ""
        loc_info = f" in {locality}" if locality else ""
        offer_info = f" using your active offer '{active_offer}'" if active_offer else ""
        body = (
            f"I don't have enough information in the supplied context to answer that accurately{store_info}{loc_info}. "
            f"Which detail should I check?"
        )
        res = {
            "action": "send",
            "body": body,
            "cta": "open_ended",
            "rationale": "States that the requested detail is absent from context and asks which detail to check."
        }
        conv_mgr.record_turn(conversation_id, "vera", res["body"], turn_number + 1)
        return res

    # 6. DEFAULT PROGRESSION
    conv_mgr.set_state(conversation_id, "progressing")
    follow_up = f"Understood {salutation}. What would you like help with next{f' for {merchant_name}' if merchant_name else ''}?"
    res = {
        "action": "send",
        "body": follow_up,
        "cta": "open_ended",
        "rationale": "Asks the merchant what help they want next without asserting that work is already complete."
    }
    conv_mgr.record_turn(conversation_id, "vera", res["body"], turn_number + 1)
    return res
