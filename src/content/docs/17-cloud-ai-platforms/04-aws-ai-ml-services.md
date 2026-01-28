---
title: "17.4 AWS AI/ML Services Stack"
---

# 17.4 AWS AI/ML Services Stack

## Introduction

Beyond Bedrock and AgentCore, AWS offers a comprehensive suite of AI/ML services including SageMaker for custom ML, and specialized AI services for vision, language, and speech. This section covers the full AWS AI/ML ecosystem and how to leverage these services effectively.

## AWS AI/ML Service Categories

- Foundation Model Services
  - Amazon Bedrock
  - SageMaker JumpStart
- Custom ML Platform
  - SageMaker Studio
  - SageMaker Training
  - SageMaker Endpoints
- Pre-trained AI Services
  - Rekognition (Vision)
  - Textract (Document)
  - Comprehend (NLP)
  - Transcribe (Speech)
  - Polly (Text-to-Speech)
  - Translate
  - Kendra (Search)

## Amazon SageMaker

### SageMaker Studio

- Unified development environment
- JupyterLab-based IDE
- Code Editor (VS Code)
- Notebook collaboration
- Git integration

### SageMaker Training

- Built-in algorithms
- Bring your own container
- Distributed training
- Spot instance training
- Hyperparameter tuning

### SageMaker Endpoints

- Real-time inference
  - Single model
  - Multi-model
  - Multi-container
- Serverless inference
  - Pay per request
  - Automatic scaling
  - Cold start consideration
- Batch transform
  - Large dataset processing
  - Scheduled jobs

### SageMaker JumpStart

- Foundation model hub
  - Llama, Falcon, Mistral
  - Stable Diffusion
  - Embedding models
- One-click deployment
- Fine-tuning support
- Solution templates

### SageMaker MLOps

- Model Registry
  - Version tracking
  - Approval workflows
  - Deployment stages
- Pipelines
  - Orchestrated workflows
  - Reproducible training
  - Automated deployment
- Model Monitor
  - Data drift detection
  - Model quality monitoring
  - Bias detection

## Amazon Rekognition

- Image analysis
  - Object detection
  - Face analysis
  - Celebrity recognition
  - Text in image
- Video analysis
  - Person tracking
  - Activity detection
  - Scene understanding
- Custom Labels
  - Custom object detection
  - Business-specific models
  - Transfer learning

## Amazon Textract

- Document processing
  - Text extraction (OCR)
  - Form extraction
  - Table extraction
- Specialized features
  - Query-based extraction
  - Expense analysis
  - Identity documents
  - Lending documents
- Asynchronous processing
  - Large document handling
  - S3 integration
  - SNS notifications

## Amazon Comprehend

- Text analytics
  - Entity recognition
  - Sentiment analysis
  - Key phrase extraction
  - Language detection
- Custom models
  - Custom classification
  - Custom entity recognition
- Comprehend Medical
  - Healthcare entities
  - PHI detection
  - Medical relationships

## Amazon Transcribe

- Speech-to-text
  - Real-time transcription
  - Batch transcription
  - Multi-language support
- Features
  - Speaker diarization
  - Custom vocabulary
  - Automatic punctuation
  - Toxicity detection
- Specialized versions
  - Transcribe Medical
  - Transcribe Call Analytics

## Amazon Polly

- Text-to-speech
  - 60+ voices
  - 30+ languages
  - Neural TTS (NTTS)
- Features
  - SSML support
  - Lexicons
  - Speech marks
  - Brand voices

## Amazon Translate

- Neural machine translation
- 75+ languages
- Features
  - Custom terminology
  - Formality control
  - Profanity masking
  - Real-time translation
  - Document translation

## Amazon Kendra

- Enterprise search
  - Semantic search
  - Natural language queries
- Features
  - 14+ data connectors
  - Access control
  - Relevance tuning
  - Custom query suggestions
- Integration
  - S3, SharePoint, Salesforce
  - Databases
  - Web crawlers

## Amazon Personalize

- ML-powered recommendations
- Real-time personalization
- Features
  - Recommendation types
  - User segmentation
  - Similar items
  - Ranking

## Amazon Forecast

- Time series forecasting
- Use cases
  - Demand planning
  - Resource planning
  - Financial planning
- Built-in algorithms
  - AutoML selection
  - DeepAR+
  - CNN-QR

## Service Integration Patterns

- Lambda + AI Services
  - Serverless processing
  - Event-driven workflows
- Step Functions orchestration
  - Multi-service pipelines
  - Error handling
  - Human approval
- EventBridge integration
  - Event-driven AI
  - Cross-service routing

## Best Practices

- Service selection
  - Pre-built vs custom
  - Cost considerations
  - Latency requirements
- Security
  - IAM policies
  - VPC endpoints
  - Encryption
- Cost optimization
  - Right-size endpoints
  - Batch processing
  - Spot instances

## Hands-on Exercises

1. Build document processing pipeline with Textract
2. Create video analysis system with Rekognition
3. Implement real-time transcription with Transcribe
4. Deploy custom model with SageMaker

## Summary

| Service | Category | Primary Use Case |
|---------|----------|------------------|
| SageMaker | Custom ML | Model training & deployment |
| Rekognition | Vision | Image/video analysis |
| Textract | Document | Text extraction |
| Comprehend | NLP | Text analytics |
| Transcribe | Speech | Speech-to-text |
| Polly | Speech | Text-to-speech |
| Kendra | Search | Enterprise search |

## Next Steps

- [Microsoft Foundry Platform](05-microsoft-foundry.md)
- [Vertex AI with Gemini](08-vertex-ai-gemini.md)
- [Performance Optimization](16-performance-optimization.md)
