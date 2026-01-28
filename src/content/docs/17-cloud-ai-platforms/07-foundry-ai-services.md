---
title: "17.7 Microsoft Foundry AI Services"
---

# 17.7 Microsoft Foundry AI Services

## Introduction

Microsoft Foundry provides a comprehensive suite of AI services beyond LLMs, including Document Intelligence, Computer Vision, Speech, Translator, and specialized vertical solutions. These services integrate seamlessly with the Foundry platform for enterprise AI applications.

## Foundry AI Services Overview

- Full suite of cognitive services
- Pre-trained models for common scenarios
- Custom model training capabilities
- Enterprise-grade SLAs
- Regional availability

## Document Intelligence

- Document processing capabilities
  - Invoice extraction
  - Receipt processing
  - ID document reading
  - Custom form extraction
- Pre-built models
  - Read (OCR + structure)
  - Layout (tables, figures)
  - General document
  - Contracts, Health records
- Custom models
  - Template-based training
  - Neural model training
  - Composed models
- 2025 updates
  - Improved accuracy
  - Better table extraction
  - Handwriting support
  - Multi-language expansion

## Computer Vision

- Image analysis capabilities
  - Object detection
  - Image classification
  - Scene understanding
  - Brand recognition
- Florence 2 multimodal model
  - Image captioning
  - Visual question answering
  - Dense region captioning
  - Object grounding
- Video analysis
  - Scene segmentation
  - Object tracking
  - Activity recognition
- Spatial analysis
  - People counting
  - Zone monitoring
  - Social distancing

## Speech Services

- Speech-to-Text
  - Real-time transcription
  - Batch transcription
  - Pronunciation assessment
  - Custom speech models
- Text-to-Speech
  - 400+ neural voices
  - Custom neural voice
  - SSML support
  - Real-time synthesis
- Speech translation
  - Multi-language support
  - Real-time translation
  - Custom terminology
- 2025 enhancements
  - HD voice quality
  - Personal voice (with consent)
  - Improved accuracy

## Translator Service

- Text translation
  - 100+ languages
  - Document translation
  - Custom translator
- Real-time translation
  - Conversation translation
  - Meeting integration
- Language customization
  - Domain-specific models
  - Terminology lists
  - Phrase dictionaries

## Content Safety

- Text moderation
  - Hate speech detection
  - Violence detection
  - Self-harm content
  - Sexual content
- Image moderation
  - Inappropriate content
  - Violent imagery
  - Custom categories
- Prompt shields
  - Jailbreak detection
  - Indirect attack protection
- Groundedness detection
  - Hallucination check
  - Source verification
- Protected material
  - Copyrighted content detection
  - Code license checking

## Azure AI Search

- Vector search capabilities
  - Embedding generation
  - Similarity search
  - Hybrid search (vector + keyword)
- Semantic ranking
  - Relevance scoring
  - Query understanding
  - Answer extraction
- Index management
  - Automatic indexing
  - Incremental updates
  - Schema management
- Integration with Foundry
  - RAG patterns
  - Agent file search
  - Knowledge bases

## Language Services

- Text analytics
  - Named entity recognition
  - Key phrase extraction
  - Sentiment analysis
  - Language detection
- Question answering
  - Custom Q&A
  - Pre-built knowledge
  - Multi-turn conversations
- Summarization
  - Extractive summary
  - Abstractive summary
  - Conversation summary
- Custom text classification
- Custom named entity recognition

## Healthcare AI

- Text Analytics for Health
  - Clinical entity extraction
  - Relation detection
  - Medical terminology
- Health Insights
  - Trial matching
  - Patient timelines
  - Cancer staging
- FHIR service integration
  - HL7 FHIR R4 support
  - Data ingestion
  - MedTech integration

## Deployment Patterns

- Multi-service container
  - Edge deployment
  - Air-gapped environments
  - On-premises scenarios
- Managed endpoints
  - Auto-scaling
  - Regional deployment
  - SLA guarantees
- Batch processing
  - Large-scale jobs
  - Async processing
  - Cost optimization

## Enterprise Features

- Data residency
  - Regional compliance
  - Sovereign cloud support
- Private networking
  - VNet integration
  - Private endpoints
- Identity and access
  - Azure AD authentication
  - Managed identities
  - RBAC

## Cost Optimization

- Commitment tiers
  - Volume discounts
  - Reserved capacity
- Batch vs real-time
  - Processing trade-offs
  - Cost comparison
- Right-sizing
  - Service tier selection
  - Throughput planning

## Hands-on Exercises

1. Build a document processing pipeline
2. Create a multi-modal content analysis system
3. Implement speech-enabled customer service
4. Design RAG with Azure AI Search

## Summary

| Service | Key Capabilities |
|---------|------------------|
| Document Intelligence | OCR, form extraction, custom models |
| Computer Vision | Florence 2, object detection, video |
| Speech | 400+ voices, custom models, translation |
| Content Safety | Moderation, jailbreak, groundedness |
| AI Search | Vector search, semantic ranking |
| Language | NER, sentiment, summarization |
| Healthcare | Clinical NLP, FHIR integration |

## Next Steps

- [Vertex AI with Gemini](08-vertex-ai-gemini.md)
- [Multi-Cloud AI Strategies](13-multi-cloud-ai.md)
- [Real-World Applications](15-real-world-applications.md)
