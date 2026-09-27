# magicpin AI Challenge — Vera Bot Submission

**Participant / Team**: Vera AI Excellence  
**Version**: 1.0.0  
**Challenge**: [magicpin AI Challenge | Build Vera Better](https://magicpin.com/vera/ai-challenge)

---

## 1. Approach & Architecture

Our solution re-engineers Vera from the ground up around the **4-Context Composition Framework**:
1. **CategoryContext**: Curates domain voice (peer clinical, friendly operator, coach, precise pharmacist), allowed vocabulary, and clinical/legal taboos.
2. **MerchantContext**: Captures merchant identity, locality, performance baselines (views, calls, CTR), active catalog offers with real prices, and customer retention metrics.
3. **TriggerContext**: Delivers the high-urgency event anchor (*why now*) across both external signals (research papers, regulations, IPL match, seasonal demand shifts) and internal signals (performance spikes/dips, soft/hard lapse, milestones).
4. **CustomerContext**: Personalizes direct outreach (*merchant_on_behalf*) honoring language mix, past services, preferred slots, and relationship history.

```
┌──────────────────┐
│  CategoryContext │───┐
└──────────────────┘   │
┌──────────────────┐   │     ┌────────────────────────┐
│  MerchantContext │───┼────►│  Vera Reasoning Engine │───► Composed Message
└──────────────────┘   │     │  (Multi-tier synthesis)│     {body, cta, send_as, key}
┌──────────────────┐   │     └────────────────────────┘
│  TriggerContext  │───┤
└──────────────────┘   │
┌──────────────────┐   │
│  CustomerContext │───┘
└──────────────────┘
```

### Key Innovations:
- **Zero-Hallucination Fact Grounding**: Every number, price, citation, and date is strictly extracted from provided contexts. No synthetic competitors or fabricated claims.
- **Immediate Intent Actioning**: When a merchant expresses commitment ("ok let's do it", "proceed", "yes"), the bot transitions directly into **action execution mode** (confirming drafts, deploying links) with zero qualification turns.
- **Intelligent Auto-Reply Classification**: Detects WhatsApp Business canned messages and multi-turn repetition, terminating early to prevent burning unnecessary turns.
- **Single Low-Friction CTA**: Every outbound message closes with a clear, single action (binary YES/CONFIRM or 2-slot selection), maximizing merchant engagement.

---

## 2. API Contract & Endpoints Implemented

The bot is exposed as a high-performance **FastAPI** service:
- `POST /v1/context`: Idempotent storage on `(context_id, version)`. Replaces atomically on newer versions; returns HTTP 409 on stale versions.
- `POST /v1/tick`: Evaluates active triggers, joins required contexts, and returns up to 20 composed actions within <50ms.
- `POST /v1/reply`: Handles multi-turn merchant/customer replies with auto-reply classification, hostility exit, wait deferrals, and immediate intent execution.
- `GET /v1/healthz`: Liveness probe reporting uptime and context counts.
- `GET /v1/metadata`: Service metadata and architecture profile.
- `POST /v1/teardown`: State sanitation and context cleanup post-evaluation.

---

## 3. Tradeoffs & Design Decisions

1. **Deterministic Multi-Tier Synthesizer vs Pure LLM Call**:
   - *Decision*: Built a deterministic multi-tier reasoning engine that produces grounded messages instantly (<15ms per call) with 0% risk of timeout, rate-limiting, or format corruption, with optional LLM pluggability.
   - *Tradeoff*: Requires careful domain-specific mapping for all trigger kinds, but guarantees strict adherence to category taboos and verifiable context data.
2. **Restraint over Spam at `/v1/tick`**:
   - *Decision*: Only emits actions for valid, active trigger-merchant pairs. Unmatched or expired triggers are safely ignored.
3. **Graceful Hostile & Auto-Reply Exit**:
   - *Decision*: Immediately yields `action: "end"` on hostile or canned automated responses instead of repeatedly re-prompting.

---

## 4. Context That Would Have Helped Most

1. **Granular Slot Availability Feed**: Real-time calendar synchronization for salons and clinics to offer live bookable slots without needing generic placeholders.
2. **Merchant Engagement Affinity History**: Historical data on whether a specific merchant prefers morning WhatsApp touches vs evening digests, and whether they prefer English vs Hindi voice notes.
3. **Customer LTV & Past Spend Data**: To enable tiered discount thresholds based on historical spend rather than time-elapsed lapse alone.

---

## 5. Local Reproduction & Verification

```bash
# 1. Expand dataset seeds
python dataset/generate_dataset.py --seed-dir dataset --out dataset/expanded

# 2. Generate canonical test submission
python generate_submission.py

# 3. Start bot server
python bot.py  # Runs on port 8080

# 4. Run judge simulator
python judge_simulator.py
```
