"""
magicpin AI Challenge — Vera Bot Submission
============================================
A state-of-the-art, deterministic, multi-context merchant engagement bot
implementing the 4-context composition framework (Category, Merchant, Trigger, Customer).

Exposes the 5 required endpoints:
- POST /v1/context
- POST /v1/tick
- POST /v1/reply
- GET  /v1/healthz
- GET  /v1/metadata
"""

from __future__ import annotations
import os
import sys
import time
import json
import re
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import conversation_handlers

app = FastAPI(title="magicpin AI Challenge - Vera Bot", version="1.0.0")
START_TIME = time.time()

# In-memory storage for contexts: (scope, context_id) -> {"version": int, "payload": dict}
contexts: Dict[tuple[str, str], Dict[str, Any]] = {}

VALID_SCOPES = {"category", "merchant", "trigger", "customer"}
MAX_PAYLOAD_BYTES = 500 * 1024  # 500 KB cap


# =============================================================================
# MIDDLEWARE: 500KB REQUEST BODY SIZE ENFORCEMENT
# =============================================================================

@app.middleware("http")
async def enforce_payload_limit(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_PAYLOAD_BYTES:
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={"error": "Payload exceeds 500KB limit"}
                )
        except ValueError:
            pass
    return await call_next(request)


# =============================================================================
# MODELS
# =============================================================================

class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: Optional[str] = None


class TickBody(BaseModel):
    now: Optional[str] = None
    available_triggers: List[str] = Field(default_factory=list)


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: Optional[str] = None
    turn_number: int = 1


# =============================================================================
# 4-CONTEXT COMPOSITION ENGINE
# =============================================================================

def format_salutation(merchant: Optional[Dict[str, Any]], category_slug: str) -> str:
    """Format merchant salutation according to category voice rules without hallucination."""
    ident = merchant.get("identity", {}) if merchant else {}
    owner = ident.get("owner_first_name", "")
    name = ident.get("name", "")

    if category_slug == "dentists":
        if owner:
            return f"Dr. {owner}"
        elif "dr" in name.lower():
            m = re.search(r"(Dr\.?\s+[A-Za-z]+)", name, re.IGNORECASE)
            return m.group(1) if m else "Doctor"
        return "Doctor"
    elif category_slug == "pharmacies":
        return owner or (name.split()[0] if name else "Partner")
    else:
        return owner or (name.split()[0] if name else "Partner")


def get_first_active_offer(merchant: Optional[Dict[str, Any]], default: str = "") -> str:
    """Extract first active offer title from merchant context."""
    if not merchant:
        return default
    offers = merchant.get("offers", [])
    for off in offers:
        if off.get("status") == "active":
            return off.get("title", "")
    return default


def prefers_hindi(obj: Optional[Dict[str, Any]]) -> bool:
    """Check if languages list contains Hindi or Hinglish preference."""
    if not obj:
        return False
    ident = obj.get("identity", {})
    langs = ident.get("languages", [])
    if isinstance(langs, list) and any(l in ["hi", "hi-en", "hindi"] for l in langs):
        return True
    lang_pref = str(ident.get("language_pref", "")).lower()
    return "hi" in lang_pref


def check_customer_consent(customer: Optional[Dict[str, Any]], trigger_kind: str) -> bool:
    """
    Validates customer consent strictly according to official challenge schema:
    customer.preferences.reminder_opt_in == True
    customer.consent.opted_in_at != None
    customer.consent.scope includes the appropriate category permission
    """
    if not customer:
        return True

    prefs = customer.get("preferences", {})
    if prefs.get("reminder_opt_in") is False:
        return False

    consent = customer.get("consent")
    if not consent or not isinstance(consent, dict):
        return False

    opted_in_at = consent.get("opted_in_at")
    if not opted_in_at:
        return False

    allowed_scopes = consent.get("scope", [])
    if not isinstance(allowed_scopes, list) or len(allowed_scopes) == 0:
        return False

    # Consent is purpose-specific: promotional permission cannot authorize a
    # service or appointment reminder. Unknown customer triggers fail closed.
    SCOPE_MATRIX = {
        "appointment_tomorrow": ["appointment_reminders"],
        "booking_reminder": ["appointment_reminders"],
        "recall_due": ["recall_reminders"],
        "recall_window_open": ["recall_reminders"],
        "chronic_refill_due": ["refill_reminders"],
        "refill_reminder": ["refill_reminders"],
        "wedding_package_followup": ["bridal_package_followup"],
        "bridal_followup": ["bridal_package_followup"],
        "customer_lapsed_hard": ["winback_offers"],
        "customer_lapsed_soft": ["recall_reminders"],
        "winback_rashmi": ["winback_offers"],
        "trial_followup": ["treatment_followup"],
    }

    target_scopes = SCOPE_MATRIX.get(trigger_kind, [])
    if not target_scopes:
        return False
    if "all" in allowed_scopes:
        return True
    return any(s in allowed_scopes for s in target_scopes)


def compose(
    category: Optional[Dict[str, Any]],
    merchant: Optional[Dict[str, Any]],
    trigger: Optional[Dict[str, Any]],
    customer: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Main 4-context composition function.
    Given CategoryContext, MerchantContext, TriggerContext, and optional CustomerContext,
    synthesizes a high-compulsion, verifiable, strictly grounded WhatsApp message.
    Returns None if required merchant/category context is absent, customer has not consented,
    or required trigger data is missing (zero invented defaults).
    """
    if not merchant or not category or not trigger:
        return None

    t_kind = trigger.get("kind", "")
    t_scope = trigger.get("scope", "merchant")
    payload = trigger.get("payload", {})
    suppression_key = trigger.get("suppression_key", f"{t_kind}:{int(time.time())}")

    # Strict customer consent check
    if t_scope == "customer" and customer is None:
        return None
    if customer or t_scope == "customer":
        if not check_customer_consent(customer, t_kind):
            return None

    cat_slug = category.get("slug", "")
    send_as = "merchant_on_behalf" if (t_scope == "customer" or customer is not None) else "vera"

    m_ident = merchant.get("identity", {})
    m_name = m_ident.get("name")
    m_locality = m_ident.get("locality")
    if not m_name:
        return None
    m_city = m_ident.get("city", "")
    salutation = format_salutation(merchant, cat_slug)
    active_offer = get_first_active_offer(merchant)
    m_perf = merchant.get("performance", {})
    m_ctr = m_perf.get("ctr")
    m_views = m_perf.get("views")
    m_calls = m_perf.get("calls")
    c_agg = merchant.get("customer_aggregate", {})

    c_ident = customer.get("identity", {}) if customer else {}
    c_name = c_ident.get("name", "there")
    use_hindi_mix = prefers_hindi(customer) if customer else prefers_hindi(merchant)

    # -------------------------------------------------------------------------
    # 1. RESEARCH DIGEST (Merchant-Facing, Category: Dentists / Medical)
    # -------------------------------------------------------------------------
    if t_kind in ["research_digest", "category_research_digest_release"]:
        top_item = payload.get("top_item", {})
        item_id = payload.get("top_item_id", "")
        digest_items = category.get("digest", [])
        for it in digest_items:
            if it.get("id") == item_id:
                top_item = it
                break

        title = top_item.get("title") or payload.get("title")
        if not title:
            return None  # Cannot compose research digest without verified title

        source = top_item.get("source") or payload.get("source", "")
        trial_n = top_item.get("trial_n") or payload.get("trial_n")
        p_segment = (top_item.get("patient_segment") or payload.get("patient_segment") or "").replace("_", " ")

        high_risk_count = c_agg.get("high_risk_adult_count")
        cohort_str = f"your {high_risk_count} {p_segment} patients" if high_risk_count and p_segment else (f"your {high_risk_count} patients" if high_risk_count else (f"your {p_segment} cohort" if p_segment else ""))
        evidence_str = f"— {trial_n:,}-patient trial showed {title}" if trial_n else f"— clinical update: {title}"

        body = (
            f"{salutation}, the latest category clinical digest landed{': ' + cohort_str if cohort_str else ''} "
            f"{evidence_str}. "
            f"Would you like a concise summary?"
            f"{f' — {source}' if source else ''}"
        )
        return {
            "body": body,
            "cta": "open_ended",
            "send_as": "vera",
            "suppression_key": suppression_key,
            "rationale": "Uses the title and any cohort, evidence count, and source present in the supplied digest."
        }

    # -------------------------------------------------------------------------
    # 2. RECALL DUE (Customer-Facing, Category: Dentists / Salons / Gyms)
    # -------------------------------------------------------------------------
    elif t_kind in ["recall_due", "recall_window_open"]:
        service = payload.get("service_due") or (customer.get("relationship", {}).get("services_received", [])[-1] if customer and customer.get("relationship", {}).get("services_received") else "")
        if not customer or not service:
            return None
        service = service.replace("_", " ")
        slots = payload.get("available_slots", [])
        slot_labels = [s.get("label") or s.get("iso") for s in slots if isinstance(s, dict) and (s.get("label") or s.get("iso"))]
        
        if len(slot_labels) >= 2:
            s1, s2 = slot_labels[:2]
            slot_text = f"Apke liye 2 slots ready hain: {s1} ya {s2}." if use_hindi_mix else f"We have 2 slots open for you: {s1} or {s2}."
        elif len(slot_labels) == 1:
            slot_text = f"Available time: {slot_labels[0]}."
        else:
            slot_text = "Please share a time that works for you."

        offer_str = f" {active_offer}" if active_offer else ""
        booking_ask = "Reply 1 or 2 to choose a slot." if len(slot_labels) >= 2 else ("Reply YES to request this time." if len(slot_labels) == 1 else "Please share a time that works for you.")
        if use_hindi_mix:
            body = (
                f"Hi {c_name}, {m_name} here. Your {service} recall is due. "
                f"{slot_text}{offer_str} {booking_ask}"
            )
        else:
            body = (
                f"Hi {c_name}, {m_name} here. Your {service} recall is due. "
                f"{slot_text}{offer_str} {booking_ask}"
            )
        return {
            "body": body,
            "cta": "open_ended",
            "send_as": "merchant_on_behalf",
            "suppression_key": suppression_key,
            "rationale": "Uses the customer relationship and any supplied appointment options; asks for a time when none are provided."
        }

    # -------------------------------------------------------------------------
    # 3. REGULATION CHANGE / COMPLIANCE (Dentists / Healthcare)
    # -------------------------------------------------------------------------
    elif t_kind in ["regulation_change", "compliance_alert"]:
        top_item = payload.get("top_item", {})
        item_id = payload.get("top_item_id", "")
        if not top_item and category:
            for it in category.get("digest", []):
                if it.get("id") == item_id:
                    top_item = it
                    break
        title = top_item.get("title") or payload.get("title")
        if not title:
            return None

        deadline = payload.get("deadline_iso", "")
        deadline_str = f" Deadline: {deadline}." if deadline and deadline not in title else ""
        body = f"{salutation}, regulatory update: {title}.{deadline_str} Would you like to review the supplied notice?"
        return {
            "body": body,
            "cta": "open_ended",
            "send_as": "vera",
            "suppression_key": suppression_key,
            "rationale": "References only the supplied regulatory title and deadline."
        }

    # -------------------------------------------------------------------------
    # 4. CHRONIC REFILL DUE (Customer-Facing, Category: Pharmacies)
    # -------------------------------------------------------------------------
    elif t_kind in ["chronic_refill_due", "refill_reminder"]:
        molecules = payload.get("molecule_list", [])
        if not customer or not molecules:
            return None
        mol_str = ", ".join(molecules)
        stock_out = payload.get("stock_runs_out_iso", "")
        out_date = stock_out.split("T")[0] if "T" in stock_out else stock_out
        if not out_date:
            return None
        date_clause = f"{out_date} ko" if use_hindi_mix else f"on {out_date}"

        disc_str = ""
        if payload.get("discount"):
            disc_str = f"{payload.get('discount')}% discount applied — "

        if use_hindi_mix:
            body = (
                f"Namaste — {m_name} {m_locality} yahan. {c_name} ji ki monthly medicines ({mol_str}) "
                f"{date_clause} khatam hongi. "
                f"{disc_str}"
                f"Reply YES if you would like to discuss this refill."
            )
        else:
            body = (
                f"Namaste — {m_name} {m_locality} here. Monthly medicines for {c_name} ({mol_str}) "
                f"run out {date_clause}. "
                f"{disc_str}"
                f"Reply YES if you would like to discuss this refill."
            )
        return {
            "body": body,
            "cta": "binary",
            "send_as": "merchant_on_behalf",
            "suppression_key": suppression_key,
            "rationale": "Uses supplied refill facts and requests a simple reply if the customer wants to discuss them."
        }

    # -------------------------------------------------------------------------
    # 5. SUPPLY / RECALL ALERT (Merchant-Facing, Category: Pharmacies)
    # -------------------------------------------------------------------------
    elif t_kind in ["supply_alert", "batch_recall"]:
        molecule = payload.get("molecule")
        if not molecule:
            return None  # Cannot compose recall alert without molecule

        batches = payload.get("affected_batches", [])
        batch_str = f" ({', '.join(batches)})" if batches else ""
        mfr = payload.get("manufacturer")
        if not mfr or not batches:
            return None
        body = (
            f"{salutation}, the trigger reports a supply alert for {molecule}{batch_str} from {mfr}. "
            f"Would you like to review the alert details?"
        )
        return {
            "body": body,
            "cta": "open_ended",
            "send_as": "vera",
            "suppression_key": suppression_key,
            "rationale": "Repeats only the recall details supplied in the trigger."
        }

    # -------------------------------------------------------------------------
    # 6. ACTIVE PLANNING INTENT (Corporate Thali, Kids Yoga, Packages)
    # -------------------------------------------------------------------------
    elif t_kind == "active_planning_intent":
        topic = payload.get("intent_topic", "")
        if "thali" in topic.lower() or cat_slug == "restaurants":
            if not topic:
                return None
            body = f"{salutation}, you mentioned {topic.replace('_', ' ')}. What prices, quantities, and delivery details should I use in a draft?"
        elif "yoga" in topic.lower() or cat_slug == "gyms":
            if not topic:
                return None
            body = f"{salutation}, you mentioned {topic.replace('_', ' ')}. What age range, schedule, and price should I use in a draft?"
        else:
            if not topic:
                return None
            pkg_name = topic.replace("_", " ").title()
            body = f"{salutation}, you mentioned {pkg_name}. What price and booking details should I use if I draft this idea?"
        return {
            "body": body,
            "cta": "open_ended",
            "send_as": "vera",
            "suppression_key": suppression_key,
            "rationale": "Requests missing planning details instead of inventing prices, schedules, or delivery terms."
        }

    # -------------------------------------------------------------------------
    # 7. IPL MATCH DAY (Restaurants)
    # -------------------------------------------------------------------------
    elif t_kind in ["ipl_match_today", "event_ipl"]:
        match = payload.get("match")
        venue = payload.get("venue")
        m_time = payload.get("match_time_iso")
        if not match:
            return None
        time_str = f" at {m_time}" if m_time else ""
        venue_str = f" at {venue}" if venue else ""

        body = (
            f"Quick heads-up {salutation} — {match}{venue_str}{time_str}. "
            f"{f'Would you like to feature your active offer {active_offer!r}?' if active_offer else 'Would you like to plan a post for the match?'}"
        )
        return {
            "body": body,
            "cta": "open_ended",
            "send_as": "vera",
            "suppression_key": suppression_key,
            "rationale": "Uses match details supplied by the trigger and asks whether the merchant wants to plan a post."
        }

    # -------------------------------------------------------------------------
    # 8. BRIDAL / WEDDING FOLLOWUP (Salons)
    # -------------------------------------------------------------------------
    elif t_kind in ["wedding_package_followup", "bridal_followup"]:
        w_date = payload.get("wedding_date", "")
        days = payload.get("days_to_wedding")
        if not customer or not (days and w_date):
            return None
        time_clause = f"{days} days until your wedding on {w_date}"

        body = (
            f"Hi {c_name} 💍 {salutation} from {m_name} {m_locality} here. "
            f"With {time_clause}, would you like to discuss bridal services?"
        )
        return {
            "body": body,
            "cta": "open_ended",
            "send_as": "merchant_on_behalf",
            "suppression_key": suppression_key,
            "rationale": "Uses the customer's supplied wedding date and countdown, and asks about bridal services."
        }

    # -------------------------------------------------------------------------
    # 9. APPOINTMENT REMINDER (Customer-Facing, Salons / Clinics)
    # -------------------------------------------------------------------------
    elif t_kind in ["appointment_tomorrow", "booking_reminder"]:
        app_time = payload.get("appointment_time")
        service_name = payload.get("service")

        # Do not claim an appointment exists without both its service and scheduled time.
        if not app_time or not service_name:
            return None

        svc_str = service_name or "appointment"
        time_str_en = f"scheduled for tomorrow at {app_time}" if app_time else "scheduled for tomorrow"
        time_str_hi = f"kal {app_time} scheduled hai" if app_time else "kal scheduled hai"

        if use_hindi_mix:
            body = (
                f"Hi {c_name}! {m_name} {m_locality} se reminder: Aapka {svc_str} appointment {time_str_hi}. "
                f"Would you like to confirm?"
            )
        else:
            body = (
                f"Hi {c_name}! Reminder from {m_name} {m_locality}: Your appointment for {svc_str} is {time_str_en}. "
                f"Would you like to confirm?"
            )
        return {
            "body": body,
            "cta": "binary",
            "send_as": "merchant_on_behalf",
            "suppression_key": suppression_key,
            "rationale": "Repeats the appointment time and service supplied by the trigger and asks whether the customer wants to confirm."
        }

    # -------------------------------------------------------------------------
    # 10. CURIOUS ASK (Salons, Restaurants, Gyms)
    # -------------------------------------------------------------------------
    elif t_kind in ["curious_ask_due", "merchant_demand_inquiry"]:
        body = f"Hi {salutation}, what service or item has been most requested this week?"
        return {
            "body": body,
            "cta": "open_ended",
            "send_as": "vera",
            "suppression_key": suppression_key,
            "rationale": "Asks the merchant for the demand detail referenced by the trigger."
        }

    # -------------------------------------------------------------------------
    # 11. WINBACK / HARD LAPSE (Gyms, Salons)
    # -------------------------------------------------------------------------
    elif t_kind in ["customer_lapsed_hard", "winback_rashmi", "winback_eligible"]:
        days_lapse = payload.get("days_since_last_visit")
        focus = payload.get("previous_focus")
        focus_str = focus.replace("_", " ") if focus else "training"

        if send_as == "merchant_on_behalf":
            if not customer or not days_lapse:
                return None
            time_clause = f"about {days_lapse} days"
            body = (
                f"Hi {c_name}, {salutation} from {m_name} here. It has been {time_clause} since your last visit. "
                f"Would you like to hear about current options for {focus_str}?"
            )
        else:
            dip_pct_val = payload.get("perf_dip_pct")
            if dip_pct_val is None:
                dip_pct_val = m_perf.get("delta_7d", {}).get("views_pct")
            if dip_pct_val is None:
                return None
            dip_pct = abs(int(dip_pct_val * 100))
            lapsed_m = payload.get("lapsed_customers_added_since_expiry") or c_agg.get("lapsed_90d_plus")
            lapsed_clause = f" and {lapsed_m} past customers crossed the recall mark" if lapsed_m else ""

            body = (
                f"{salutation}, your profile views dipped {dip_pct}%{lapsed_clause}. "
                f"Would you like to discuss a customer re-engagement message?"
            )
        return {
            "body": body,
            "cta": "binary",
            "send_as": send_as,
            "suppression_key": suppression_key,
            "rationale": "References supplied lapse and performance context and asks whether the merchant wants to discuss re-engagement."
        }

    # -------------------------------------------------------------------------
    # 12. SOFT LAPSE (Healthcare / Salons / Pharmacies)
    # -------------------------------------------------------------------------
    elif t_kind in ["customer_lapsed_soft", "lapse_warning"]:
        if send_as == "merchant_on_behalf":
            offer_str = active_offer
            offer_clause = f" featuring {offer_str}" if offer_str else ""
            if use_hindi_mix:
                body = (
                    f"Hi {c_name}, {m_name} {m_locality} se check-in. "
                    f"Would you like information about your next visit{offer_clause}?"
                )
            else:
                body = (
                    f"Hi {c_name}, a check-in from {m_name} {m_locality}. "
                    f"Would you like information about your next visit{offer_clause}?"
                )
        else:
            lapsed_count = c_agg.get("lapsed_180d_plus") or c_agg.get("lapsed_90d_plus")
            if lapsed_count is None:
                return None
            body = (
                f"{salutation}, {lapsed_count} customers in your roster have crossed the soft-lapse window in {m_locality}. "
                f"Would you like to discuss a recall message{f' featuring {active_offer}' if active_offer else ''}?"
            )
        return {
            "body": body,
            "cta": "binary",
            "send_as": send_as,
            "suppression_key": suppression_key,
            "rationale": "Uses the merchant-provided lapse count where available and asks about a relevant next step."
        }

    # -------------------------------------------------------------------------
    # 13. COMPETITOR OPENED (Local Commerce)
    # -------------------------------------------------------------------------
    elif t_kind in ["competitor_opened", "local_competition"]:
        comp_name = payload.get("competitor_name")
        if not comp_name:
            return None

        dist = payload.get("distance_km")
        dist_clause = f" opened {dist}km away" if dist is not None else ""
        comp_offer = payload.get("their_offer")
        offer_clause = f" promoting {comp_offer}" if comp_offer else ""

        body = (
            f"{salutation}, the trigger reports that {comp_name}{dist_clause}{offer_clause}. "
            f"Would you like to review your listing and decide whether to update it?"
        )
        return {
            "body": body,
            "cta": "open_ended",
            "send_as": "vera",
            "suppression_key": suppression_key,
            "rationale": "References only the competitor details supplied by the trigger."
        }

    # -------------------------------------------------------------------------
    # 14. PERFORMANCE DIP (Internal Operations)
    # -------------------------------------------------------------------------
    elif t_kind in ["perf_dip", "traffic_dip"]:
        metric = payload.get("metric")
        if not metric:
            return None
        delta_val = payload.get("delta_pct")
        if delta_val is None:
            d7 = m_perf.get("delta_7d", {})
            delta_val = d7.get(f"{metric}_pct")
        if delta_val is None:
            return None  # Cannot claim performance dip without delta

        if not isinstance(delta_val, (int, float)):
            return None
        delta = abs(int(delta_val * 100))
        baseline = payload.get("vs_baseline") or m_perf.get(metric)
        baseline_clause = f" (vs {baseline} weekly baseline)" if baseline else ""
        window_clause = f" over {payload['window']}" if payload.get("window") else ""

        body = (
            f"{salutation}, your {metric} dropped {delta}%{window_clause}{baseline_clause}. "
            f"Would you like to review recent listing changes that might explain the drop?"
        )
        return {
            "body": body,
            "cta": "open_ended",
            "send_as": "vera",
            "suppression_key": suppression_key,
            "rationale": "References the supplied performance change and asks whether the merchant wants to review it."
        }

    # -------------------------------------------------------------------------
    # 15. PERFORMANCE SPIKE (Internal Operations)
    # -------------------------------------------------------------------------
    elif t_kind in ["perf_spike", "traffic_surge"]:
        metric = payload.get("metric")
        if not metric:
            return None
        delta_val = payload.get("delta_pct")
        if delta_val is None:
            d7 = m_perf.get("delta_7d", {})
            delta_val = d7.get(f"{metric}_pct")
        if delta_val is None:
            return None

        if not isinstance(delta_val, (int, float)):
            return None
        delta = int(delta_val * 100)
        driver = payload.get("likely_driver")
        driver_clause = f"; the trigger attributes it to {driver.replace('_', ' ')}" if driver else ""
        window_clause = f" over {payload['window']}" if payload.get("window") else ""

        body = (
            f"{salutation}, your {metric} rose {delta}%{window_clause}{driver_clause}. "
            f"Would you like to discuss a next step?"
        )
        return {
            "body": body,
            "cta": "open_ended",
            "send_as": "vera",
            "suppression_key": suppression_key,
            "rationale": "References the supplied performance change and any supplied likely driver."
        }

    # -------------------------------------------------------------------------
    # 16. SEASONAL PERFORMANCE DIP (Gyms / Retail)
    # -------------------------------------------------------------------------
    elif t_kind in ["seasonal_perf_dip", "seasonal_lull"]:
        delta_val = payload.get("delta_pct")
        if delta_val is None:
            delta_val = m_perf.get("delta_7d", {}).get("views_pct")
        if not isinstance(delta_val, (int, float)):
            return None
        delta = abs(int(delta_val * 100))
        active_members = c_agg.get("total_unique_ytd")
        members_clause = f"your {active_members} customers" if active_members else "your customer base"
        window_clause = f" over {payload['window']}" if payload.get("window") else ""

        body = (
            f"{salutation}, your profile views are down {delta}%{window_clause}. "
            f"Would you like to review the trend and discuss options for {members_clause}?"
        )
        return {
            "body": body,
            "cta": "open_ended",
            "send_as": "vera",
            "suppression_key": suppression_key,
            "rationale": "References the supplied performance change and asks whether the merchant wants to review it."
        }

    # -------------------------------------------------------------------------
    # 17. CATEGORY SEASONAL DEMAND SHIFT (Pharmacies / Retail)
    # -------------------------------------------------------------------------
    elif t_kind in ["category_seasonal", "summer_demand"]:
        trends = payload.get("trends", [])
        if not trends:
            return None  # Cannot claim seasonal shift without explicit trends data

        trend_summary = ", ".join(trends[:3]).replace("_", " ")
        body = (
            f"{salutation}, the supplied trend signals are: {trend_summary}. "
            f"Would you like to discuss whether these trends are relevant to your business?"
        )
        return {
            "body": body,
            "cta": "open_ended",
            "send_as": "vera",
            "suppression_key": suppression_key,
            "rationale": "References only the trend signals supplied with the trigger."
        }

    # -------------------------------------------------------------------------
    # 18. FESTIVAL UPCOMING (All Categories)
    # -------------------------------------------------------------------------
    elif t_kind in ["festival_upcoming", "festive_season"]:
        fest = payload.get("festival")
        days = payload.get("days_until")
        if not fest or days is None:
            return None

        body = (
            f"{salutation}, {fest} is coming up in {days} days. "
            f"Would you like to plan a promotion{f' featuring {active_offer}' if active_offer else ''}?"
        )
        return {
            "body": body,
            "cta": "open_ended",
            "send_as": "vera",
            "suppression_key": suppression_key,
            "rationale": "Uses the supplied festival and countdown without assuming demand or capacity."
        }

    # -------------------------------------------------------------------------
    # 19. MILESTONE REACHED (All Categories)
    # -------------------------------------------------------------------------
    elif t_kind in ["milestone_reached", "review_milestone"]:
        val_now = payload.get("value_now")
        m_target = payload.get("milestone_value")
        if val_now is None or m_target is None:
            return None

        metric = payload.get("metric")
        if metric and "review" not in str(metric).lower():
            return None
        gap = m_target - val_now
        metric_label = "reviews" if metric and "review" in str(metric).lower() else "the reported metric"
        body = (
            f"{salutation}, {m_name} is at {val_now} {metric_label}, {gap} from {m_target}. "
            f"Would you like to mark this milestone?"
        )
        return {
            "body": body,
            "cta": "open_ended",
            "send_as": "vera",
            "suppression_key": suppression_key,
            "rationale": "References the supplied milestone values without adding unsupported rankings."
        }

    # -------------------------------------------------------------------------
    # 20. UNVERIFIED GBP PROFILE (Pharmacies / Clinics / Salons)
    # -------------------------------------------------------------------------
    elif t_kind in ["gbp_unverified", "profile_verification"]:
        uplift_val = payload.get("estimated_uplift_pct")
        if payload.get("verified") is not False:
            return None
        uplift_str = f" The trigger includes an estimated uplift of {int(uplift_val * 100)}%." if uplift_val is not None else ""
        body = (
            f"{salutation}, the context says your Google listing is unverified.{uplift_str} "
            f"Would you like to review the verification options?"
        )
        return {
            "body": body,
            "cta": "binary",
            "send_as": "vera",
            "suppression_key": suppression_key,
            "rationale": "References the verification status supplied by the trigger without inventing impact or completion time."
        }

    # -------------------------------------------------------------------------
    # 21. CDE / WEBINAR / PROFESSIONAL ADVANCEMENT (Dentists)
    # -------------------------------------------------------------------------
    elif t_kind in ["cde_opportunity", "webinar_alert"]:
        credits = payload.get("credits")
        credits_str = f"{credits} credits" if credits is not None else ""
        title = payload.get("title")
        digest_item_id = payload.get("digest_item_id")
        if not title and digest_item_id:
            title = next((item.get("title") for item in category.get("digest", []) if item.get("id") == digest_item_id), None)
        if not title:
            return None
        date_str = payload.get("event_date") or payload.get("date_iso")
        date_clause = f"Event date: {date_str}. " if date_str else ""

        body = (
            f"{salutation}, the supplied context lists '{title}'{f' with {credits_str}' if credits_str else ''}. {date_clause}"
            f"Would you like to review its registration details?"
        )
        return {
            "body": body,
            "cta": "binary",
            "send_as": "vera",
            "suppression_key": suppression_key,
            "rationale": "References the webinar title and any supplied date or credit count."
        }

    # -------------------------------------------------------------------------
    # 22. DORMANCY WITH VERA (All Categories)
    # -------------------------------------------------------------------------
    elif t_kind in ["dormant_with_vera", "inactivity_check"]:
        days = payload.get("days_since_last_merchant_message")
        if days is not None:
            body = (
                f"Hi {salutation}, it has been {days} days since our last recorded message. "
                f"Would you like to review your listing or share a current priority?"
            )
            rationale = "References the supplied time since last merchant message."
        else:
            # The dormancy interval is absent in some canonical triggers. A
            # factual merchant performance snapshot can still support a useful
            # check-in without claiming how long the merchant has been inactive.
            window = m_perf.get("window_days")
            views = m_perf.get("views")
            calls = m_perf.get("calls")
            if not isinstance(window, (int, float)) or not any(
                isinstance(value, (int, float)) for value in (views, calls)
            ):
                return None
            facts = []
            if isinstance(views, (int, float)):
                facts.append(f"{views:,} views")
            if isinstance(calls, (int, float)):
                facts.append(f"{calls:,} calls")
            body = (
                f"Hi {salutation}, your listing recorded {' and '.join(facts)} over the last "
                f"{window:g} days. Would you like to review your listing or share a current priority?"
            )
            rationale = "Uses the supplied listing performance snapshot without inferring a dormancy interval."
        return {
            "body": body,
            "cta": "open_ended",
            "send_as": "vera",
            "suppression_key": suppression_key,
            "rationale": rationale
        }

    # -------------------------------------------------------------------------
    # 23. REVIEW THEME EMERGED (Restaurants / Salons / Healthcare)
    # -------------------------------------------------------------------------
    elif t_kind in ["review_theme_emerged", "sentiment_alert"]:
        theme = payload.get("theme")
        if not theme:
            return None  # Cannot warn on review theme without theme

        theme_clean = theme.replace("_", " ")
        count = payload.get("occurrences_30d")
        count_clause = f"{count} recent reviews mention" if count is not None else "Recent reviews mention"
        ctr = m_perf.get("ctr")
        ctr_clause = f" The supplied CTR is {ctr * 100:.1f}%." if isinstance(ctr, (int, float)) else ""
        body = (
            f"{salutation}, {count_clause} '{theme_clean}' at {m_name}.{ctr_clause} "
            f"Would you like help drafting a response?"
        )
        return {
            "body": body,
            "cta": "open_ended",
            "send_as": "vera",
            "suppression_key": suppression_key,
            "rationale": "References the supplied review theme and count, when provided."
        }

    # -------------------------------------------------------------------------
    # 24. TRIAL FOLLOWUP (Gyms / Salons)
    # -------------------------------------------------------------------------
    elif t_kind in ["trial_followup", "trial_session_complete"]:
        opts = payload.get("next_session_options", [])
        if not opts or not customer:
            return None
        opt_label = opts[0].get("label") or opts[0].get("iso")
        if not opt_label:
            return None
        body = (
            f"Hi {c_name}, the next-session option in the supplied context is {opt_label}. "
            f"Would you like to confirm it?"
        )
        return {
            "body": body,
            "cta": "open_ended",
            "send_as": "merchant_on_behalf",
            "suppression_key": suppression_key,
            "rationale": "Uses the next-session option supplied with the trigger."
        }

    # -------------------------------------------------------------------------
    # 25. RENEWAL DUE (All Categories)
    # -------------------------------------------------------------------------
    elif t_kind in ["renewal_due", "subscription_expiring"]:
        days_rem = payload.get("days_remaining") or merchant.get("subscription", {}).get("days_remaining")
        if days_rem is None:
            return None

        plan = payload.get("plan") or merchant.get("subscription", {}).get("plan")
        if not plan:
            return None
        amount = payload.get("renewal_amount")
        amount_clause = f" at ₹{amount:,}" if amount else ""
        views_clause = f" The context records {m_views:,} profile views" if isinstance(m_views, (int, float)) else ""
        calls_clause = f" and {m_calls:,} direct customer calls" if isinstance(m_calls, (int, float)) else ""

        body = (
            f"{salutation}, your Vera {plan} subscription has {days_rem} days remaining. "
            f"{views_clause}{calls_clause}. "
            f"Would you like to review the renewal details{amount_clause}?"
        )
        return {
            "body": body,
            "cta": "binary",
            "send_as": "vera",
            "suppression_key": suppression_key,
            "rationale": "Uses only the subscription facts present in the supplied context."
        }

    # -------------------------------------------------------------------------
    # 26. UNKNOWN TRIGGER: Omit action (Do not hallucinate ungrounded generic message)
    # -------------------------------------------------------------------------
    else:
        return None


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/v1/healthz")
async def healthz():
    """Liveness probe reporting uptime and context inventory."""
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _), _ in contexts.items():
        counts[scope] = counts.get(scope, 0) + 1
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts
    }


@app.get("/v1/metadata")
async def metadata():
    """Returns bot identity, architecture, and team credentials."""
    return {
        "team_name": "Vera AI Team",
        "team_members": ["Engineering"],
        "model": "hybrid-deterministic-reasoning-engine",
        "approach": "Deterministic 4-context synthesis engine with multi-turn intent transition, auto-reply classification, and zero-hallucination grounding",
        "contact_email": "vera-ai-challenge@magicpin.in",
        "version": "1.0.0",
        "submitted_at": datetime.now(timezone.utc).isoformat()
    }


@app.post("/v1/context")
async def push_context(body: CtxBody):
    """
    Receive context pushes from judge harness.
    Idempotent by (context_id, version).
    STRICT RULE: If incoming version <= existing version, MUST return HTTP 409 Conflict without modifying state.
    Only strictly newer versions (version > current_version) are accepted.
    """
    if body.scope not in VALID_SCOPES:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"accepted": False, "reason": f"Invalid scope '{body.scope}'. Must be one of {list(VALID_SCOPES)}"}
        )

    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] >= body.version:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "accepted": False,
                "reason": "stale_version",
                "current_version": cur["version"],
                "incoming_version": body.version
            }
        )

    contexts[key] = {
        "version": body.version,
        "payload": body.payload
    }
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.now(timezone.utc).isoformat()
    }


@app.post("/v1/tick")
async def tick(body: TickBody):
    """
    Periodic wake-up endpoint.
    Evaluates active triggers and composes outbound messages up to the 20 actions cap.
    Checks trigger expiry against 'now' and fails closed on malformed timestamps or expired events.
    """
    now_dt = None
    if body.now:
        try:
            now_dt = datetime.fromisoformat(body.now.replace("Z", "+00:00"))
        except Exception:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": f"Invalid or malformed 'now' timestamp: '{body.now}'"}
            )
    else:
        now_dt = datetime.now(timezone.utc)

    actions = []
    triggers_to_process = body.available_triggers[:50]

    for trg_id in triggers_to_process:
        if len(actions) >= 20:
            break

        trg_entry = contexts.get(("trigger", trg_id))
        trg = trg_entry.get("payload") if trg_entry else None
        if not trg:
            continue

        # Check trigger expiry strictly (fails closed)
        exp_str = trg.get("expires_at")
        if exp_str:
            try:
                exp_dt = datetime.fromisoformat(exp_str.replace("Z", "+00:00"))
                if now_dt >= exp_dt:
                    continue  # Strictly discard expired trigger
            except Exception:
                continue  # Fail closed on corrupt expiry timestamp

        merchant_id = trg.get("merchant_id")
        m_entry = contexts.get(("merchant", merchant_id)) if merchant_id else None
        merchant = m_entry.get("payload") if m_entry else None

        # Omit action if merchant context is missing
        if not merchant:
            continue

        cat_slug = merchant.get("category_slug", "") if merchant else trg.get("payload", {}).get("category", "")
        cat_entry = contexts.get(("category", cat_slug)) if cat_slug else None
        category = cat_entry.get("payload") if cat_entry else None

        # Omit action if category context is missing
        if not category:
            continue

        customer_id = trg.get("customer_id")
        c_entry = contexts.get(("customer", customer_id)) if customer_id else None
        customer = c_entry.get("payload") if c_entry else None
        if customer_id and customer is None:
            continue

        # Compose message with strict grounding & consent enforcement
        composed = compose(category=category, merchant=merchant, trigger=trg, customer=customer)
        if not composed:
            continue

        actions.append({
            "conversation_id": f"conv_{merchant_id}_{trg_id}",
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": composed["send_as"],
            "trigger_id": trg_id,
            "template_name": f"vera_{cat_slug}_v1" if cat_slug else "vera_generic_v1",
            "template_params": [
                merchant.get("identity", {}).get("name", "Merchant"),
                trg.get("kind", "update")
            ],
            "body": composed["body"],
            "cta": composed["cta"],
            "suppression_key": composed["suppression_key"],
            "rationale": composed["rationale"]
        })

    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    """
    Multi-turn conversation reply endpoint.
    Dispatches to conversation_handlers.respond() to handle auto-replies,
    commitments, hostility, or general questions with terminal state enforcement.
    """
    m_entry = contexts.get(("merchant", body.merchant_id)) if body.merchant_id else None
    merchant = m_entry.get("payload") if m_entry else None

    cat_slug = merchant.get("category_slug", "") if merchant else ""
    cat_entry = contexts.get(("category", cat_slug)) if cat_slug else None
    category = cat_entry.get("payload") if cat_entry else None

    c_entry = contexts.get(("customer", body.customer_id)) if body.customer_id else None
    customer = c_entry.get("payload") if c_entry else None

    resp = conversation_handlers.respond(
        conversation_id=body.conversation_id,
        merchant_id=body.merchant_id,
        customer_id=body.customer_id,
        from_role=body.from_role,
        message=body.message,
        turn_number=body.turn_number,
        merchant=merchant,
        category=category,
        customer=customer
    )
    return resp


@app.post("/v1/teardown")
async def teardown():
    """Wipes in-memory contexts and conversation states upon test completion."""
    contexts.clear()
    conversation_handlers.conv_mgr.clear()
    return {"status": "cleared", "ts": datetime.now(timezone.utc).isoformat()}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("bot:app", host="0.0.0.0", port=port, reload=False)
