Backend API – Recommendation Service

This repository contains the Spring Boot backend service, which acts as the system boundary for the recommendation platform.

The backend owns API contracts, integrates with the ML inference service, and guarantees deterministic behavior under all conditions.

Responsibilities

Validate incoming requests

Call the ML inference service

Enforce cold-start and fallback logic

Guarantee a valid response to consumers

Annotate responses with explainability metadata

Core Behavior

When a recommendation request is received:

Backend calls the inference service

If ML returns valid recommendations → return them

If ML returns empty results or errors → return fallback recommendations

Response includes:

source: ml or fallback

reason (if fallback is used)

This ensures the system never fails silently.

Example Response
{
  "user_id": 1,
  "recommendations": [1, 2, 3, 101, 102],
  "source": "fallback_popular",
  "reason": "empty_recs_or_cold_start"
}

Technology

Java

Spring Boot

REST APIs

Docker

AWS
