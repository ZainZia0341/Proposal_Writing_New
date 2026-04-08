# Professional Portfolio - GeeksVisor

## Table of Contents
- [Full Stack / Backend Projects](#full-stack--backend-projects)
- [AI & GenAI Projects](#ai--genai-projects)
- [Bubble.io Projects](#bubbleio-projects)
- [Landing Pages & Websites](#landing-pages--websites)
- [UI/UX Design](#uiux-design)
- [Project Videos](#project-videos)

---

## Full Stack / Backend Projects

### EarlyBirdee
**Role:** Full Stack Developer

**Description:** Job search platform with AI-powered matching

**Key Achievements:**
- Developed serverless architecture for daily job fetching from multiple ATS
- Implemented generative AI for job filtering and personalized matching
- Built subscription-based model with automated user notifications
- Integrated Slack for team communications and alerting

**Technologies:** React, Redux, TypeScript, Node.js, AWS Lambda, DynamoDB, AWS AppSync, API Gateway, AWS Step Functions, SQS, SNS, CloudWatch, Generative AI, GitHub Actions

---

### SaddleFit
**Role:** Full Stack Developer

**Description:** Multi-tenant e-commerce for saddle fitting & sales

**Key Achievements:**
- Designed serverless backend with custom CMS & transaction workflows
- Synced products/orders with Shopify; handled payments with Stripe
- Automated shipping using Shipo; booking via Calendly
- Built buyer/seller dashboards with real-time tracking
- Implemented CI/CD for rapid, reliable deployments

**Technologies:** React (Vite), TypeScript, Node.js, AWS Lambda, API Gateway, DynamoDB, Cognito, Step Functions, EventBridge, SQS, SNS, CloudWatch, Shopify, Stripe, Shipo, CodePipeline/CodeBuild

---

### SoPlan
**Role:** Backend Developer

**Description:** Online appointment scheduling with calendar & payments

**Key Achievements:**
- Built entire serverless backend from scratch
- Integrated Google/Office365 calendars and Zoom meetings
- Implemented Stripe subscriptions and billing
- Set up production CI/CD pipelines

**Technologies:** TypeScript, Node.js, Next.js, AWS Lambda, AppSync (GraphQL), DynamoDB, Cognito, CloudFormation, IAM, Secrets Manager, SSM Parameter Store, CloudWatch, CodePipeline/Build/Deploy

---

### Logistics Platform
**Role:** Full Stack Developer

**Description:** Trade compliance & global logistics multi-tenant SaaS

**Key Achievements:**
- Architected multi-tenant platform with fine-grained auth (OpenFGA)
- Automated product classification (HTS/ECCN/Schedule B) and compliance checks
- Orchestrated shipments, tracking, and exception handling (event-driven)
- Delivered dashboards for transaction visibility and lifecycle management

**Technologies:** React, TypeScript, NestJS, Node.js, AWS Lambda, DynamoDB (single-table), S3, API Gateway, Cognito, OpenFGA, EventBridge, Step Functions, SQS, SNS, CloudWatch/X-Ray, CDK Pipelines, GitHub Actions

---

### Qwibbs
**Role:** Full Stack Developer

**Description:** Gift-Funds and Events platform

**Key Achievements:**
- Create a platform where users can create gifts & events
- Contribute to gift funds created by linked users
- Notify users about gift fund creation by linked users
- Serverless architecture with AWS Amplify

**Technologies:** Node.js, AWS Lambda, Cognito, DynamoDB, S3, CloudWatch, Step Functions, SQS, SNS, API Gateway, AWS AppSync, EventBridge, OpenSearch, Next.js, TypeScript, Tailwind CSS, GitLab CI/CD

---

### ViralApp
**Role:** Backend Developer

**Description:** Social content scraping & insights for TikTok/Instagram

**Key Achievements:**
- Implemented scraping via RapidAPI and orchestrated pipelines with Step Functions
- Generated thumbnails and stored media in S3; exposed optimized data APIs
- Reduced Bubble.io integration latency from ~1s to ~150ms

**Technologies:** Node.js, AWS Lambda, Step Functions, API Gateway, S3, RDS, EC2, CloudWatch, RapidAPI, Bubble.io

---

### Zapyan
**Role:** Backend Developer (Lead)

**Description:** Recommendation platform with hyperscale serverless backend

**Key Achievements:**
- Built event-driven microservices integrating multiple data sources (incl. Shopify)
- Used provisioned concurrency to stabilize latency on public APIs
- Operated at millions of requests with high availability
- Ran workshops and instituted cost controls (off-hours environment shutdown)

**Technologies:** Node.js, TypeScript, AWS Lambda, DynamoDB, AppSync (GraphQL), API Gateway, Step Functions, EventBridge, Neptune, CloudWatch

---

### Deepbloo
**Role:** Backend Developer

**Description:** Energy sales & market intelligence platform

**Key Achievements:**
- Migrated monolith to microservices using the Strangler Fig pattern
- Built GraphQL layer with AppSync and mixed resolvers (RDS/Lambda/HTTP)
- Tuned PostgreSQL performance and automated infra with AWS CDK

**Technologies:** API Gateway, AWS Lambda, AWS AppSync, Cognito, RDS (PostgreSQL), AWS CDK, CloudWatch

---

### Real-Time Trade Alert Application
**Role:** Full Stack Developer

**Description:** Real-time trade alerts with GraphQL subscriptions

**Key Achievements:**
- Fully serverless backend built with AWS Serverless Framework
- AWS AppSync for GraphQL API (queries + real-time subscriptions)
- Custom Admin Dashboard for creating/pushing trade alerts
- Authentication flows with Cognito hosted UI + JWT tokens

**Technologies:** Next.js, AWS AppSync, AWS Lambda, DynamoDB, Cognito, Serverless Framework, Apollo Client, GitHub Actions

---

## AI & GenAI Projects

### LedgerIQ.ai
**Role:** Full Stack / GenAI Developer

**Description:** AI-driven bookkeeping with conversational analytics

**Key Features:**
- Built RAG pipelines for FAQ and user-document retrieval (MongoDB vector + Titan embeddings)
- Converted natural-language questions into structured MongoDB queries
- Categorized conversations for admin reporting; added learning suggestions (YouTube/blog)
- Deployed on AWS Fargate/ECS behind API Gateway; agentic flows with LangGraph

**Technologies:** Vue.js, FastAPI, Mistral-large-latest, AWS Bedrock (Titan embeddings), MongoDB vector, pandas, matplotlib, ReportLab, AWS Fargate, ECS, API Gateway, LangChain, LangGraph

---

### Aha-doc
**Role:** Full Stack Developer

**Description:** Collaborative document analysis platform with AI-powered citations

**Key Features:**
- Upload documents and have AI-powered conversations based on their content
- Organize documents into public or private workspaces
- Source-backed answers with clickable citations that jump to exact page and highlight text
- OCR support for scanned documents

**Technologies:** Vue.js, PDF.js, Node.js, PostgreSQL, PyMuPDF (fitz), Tesseract (OCR), Pinecone, OpenAI embedding model, GPT-4o-mini

---

### Contextufy
**Role:** Full Stack Developer

**Description:** AI-powered file management and collaboration application

**Key Features:**
- Secure file upload with encryption in S3 and metadata in DynamoDB
- AI chatbot searches file metadata to suggest relevant documents
- Users select files as context for AI to generate insights
- Personal vault and collaborative groups/channels for sharing

**Technologies:** Next.js, Node.js, AWS Lambda, DynamoDB, S3, LangChain, LangGraph, FastAPI, Llama 4 Scout (4-bit quantized), AWS SageMaker, ECR

---

### A+ Resumes
**Role:** AI Developer

**Description:** AI-powered resume improvement tool

**Key Features:**
- Instant, personalized feedback on resume content, formatting, and writing quality
- Custom 3-point grading system
- Users upload PDF and get clear suggestions to improve
- Helps stand out to hiring managers

**Technologies:** Python, OpenAI API, PDF processing, NLP

---

### Chatbot with RAG Capabilities
**Role:** AI Developer

**Description:** RAG-based chatbot using Claude-3-Sonnet

**Key Features:**
- Developed sophisticated RAG chatbot using Streamlit and Anthropic Claude-3-Sonnet
- Leverages semantic search for contextually accurate responses
- FAISS for in-memory vector storage
- Bedrock embedding model for efficient text retrieval

**Technologies:** Streamlit, Anthropic Claude-3-Sonnet, AWS Bedrock, FAISS, boto3, LangChain

---

### MultiModal Chatbot
**Role:** AI Developer

**Description:** Interactive multimodal chatbot with text and image inputs

**Key Features:**
- Engage in conversations by submitting text inputs or uploading images
- Processes inputs and responds using generative AI
- User-friendly interactive interface

**Technologies:** Streamlit, AWS Bedrock, Multimodal processing

---

### Jokes from News Using LLM
**Role:** AI Developer

**Description:** AI application generating jokes from news headlines

**Key Features:**
- Generates humorous jokes from news headlines
- Users input news articles and receive jokes
- Feedback system for model fine-tuning
- Integration with Hugging Face's FLAN-T5 model

**Technologies:** Streamlit, Hugging Face Transformers, PEFT, LoRA, LangChain, JSON storage

---

### PEFT with PPO to Minimize Hate Speech
**Role:** AI/ML Developer

**Description:** Fine-tuning FLAN-T5 with reinforcement learning to reduce toxicity

**Key Features:**
- Fine-tunes FLAN-T5 model using Reinforcement Learning (PPO) and PEFT
- Generates less-toxic summaries
- Uses Meta AI's hate speech reward model
- Evaluates model performance quantitatively and qualitatively

**Technologies:** Hugging Face Transformers, PyTorch, TRL, PEFT, LoRA, PPO, pandas

---

## Bubble.io Projects

### Homeworke
**Role:** Bubble Developer
**Website:** www.homeworke.com

**Description:** All-in-one home services platform (largest project, ongoing support)

**Key Features:**
- Find trusted service experts, compare prices, book appointments
- Full home renovations, cleaning services, real estate property valuation
- Project management with vetted pros (licensed, bonded, insured)
- In-app messaging, SMS/email notifications, calendar integration
- Role-based dashboards for customers, pros, and admins

**Technologies:** Bubble.io, External APIs, Payment integrations

---

### NextPlayToday
**Role:** Bubble Developer
**Website:** http://nextplaytoday.com

**Description:** Student-athlete tracking and recruitment platform

**Key Features:**
- Track progress, enhance skills with top trainers
- Identify the right college fit
- AI comparison to real Division 1 players
- Personalized training recommendations
- Helps athletes get recruited and win scholarships

**Technologies:** Bubble.io, AI integration

---

### Imentr
**Role:** Bubble Developer
**Website:** https://imentr.com/version-test

**Description:** Platform with booking feature (in finalization)

**Key Features:**
- Finalizing booking feature
- User management and scheduling

**Technologies:** Bubble.io, Calendar APIs

---

### ViralApp
**Role:** Bubble Developer
**Website:** go.viralapp.io

**Description:** Social content scraping for TikTok and Instagram (offline by client)

**Key Features:**
- Scrape posts and reels from famous content creators
- Analyze viral content patterns
- Help users align content strategies to increase viral chances

**Technologies:** Bubble.io, Node.js, AWS Lambda, RapidAPI, S3, RDS, EC2

---

### CartRentalDriver
**Role:** Bubble Developer
**Website:** https://cartrentaldriver.bubbleapps.io/version-test

**Description:** Electric vehicle rental platform

**Key Features:**
- Rent electric vehicles
- User and vehicle management

**Technologies:** Bubble.io

---

### Beaglez
**Role:** Bubble Developer
**Website:** https://app.beaglez.co.za

**Description:** Business application

**Key Features:**
- Business management features

**Technologies:** Bubble.io

---

### Gross Margin Calculator
**Role:** Bubble Developer
**Website:** https://gross-margin-calculator.bubbleapps.io/version-test/signin

**Description:** Financial calculation tool

**Key Features:**
- Calculate gross margins
- Financial analytics

**Technologies:** Bubble.io

---

## Landing Pages & Websites

### SaddleFit
- **Website:** saddlefit.io
- **Type:** Landing Page + Full Application

### WealthBuilder
- **Website:** wealthbuilder.io
- **Type:** Landing Page

### A+ Resumes
- **Website:** aplusresumes.ai
- **Type:** Landing Page + AI Tool

### KyoGreen
- **Website:** kyogreen.com
- **Type:** Landing Page

### Mifu
- **Website:** mifu.co.uk
- **Type:** Landing Page

### Salvesen
- **Website:** salvesen.app
- **Type:** Landing Page

### GeeksVisor
- **Website:** geeksvisor.com
- **Type:** Landing Page

### Qwibbs
- **Website:** qwibbs.com
- **Type:** Landing Page

---

## UI/UX Design

**Figma Portfolio:**
- [Design Projects Portfolio](https://www.figma.com/design/mJCkLkVupmLbhLZSMUTsx0/Design-Projects?node-id=0-1&t=esGG99F1OnZOeADd-1)

---

## Project Videos

**Demo Videos:**
1. [LedgerIQ Demo](https://www.loom.com/share/1f3ceeea56ec46c5ada7046f06d3e909?sid=5dd05a1e-5ce9-492a-8dc9-891b30baab89)
2. [Project Demo 2](https://www.loom.com/share/53ab4c5e6d4548b1bd146ce9ea701a1c?sid=8a5fdbdd-799c-4548-8810-e96210a3c0b0)
3. [Project Demo 3](https://www.loom.com/share/5e9e71914586459bb7d249cea6563f16?sid=449d49b2-0c8e-4d3c-a591-726d0d46a01b)

---

## Technologies & Skills Summary

### Cloud & Backend
- **AWS Services:** Lambda, DynamoDB, S3, API Gateway, AppSync, Cognito, Step Functions, EventBridge, SQS, SNS, CloudWatch, Fargate, ECS
- **Serverless Frameworks:** Serverless Framework, AWS CDK, AWS SAM
- **Databases:** DynamoDB, PostgreSQL, MongoDB, RDS
- **APIs:** GraphQL (AppSync), REST, WebSockets

### Frontend & UI
- **Frameworks:** React.js, Next.js, Vue.js, Vite
- **State Management:** Redux, Apollo Client
- **Styling:** Tailwind CSS, Material UI
- **No-Code:** Bubble.io

### AI & Machine Learning
- **LLM Integration:** OpenAI (GPT-4), Anthropic Claude, AWS Bedrock, Mistral
- **AI Frameworks:** LangChain, LangGraph
- **Vector Databases:** Pinecone, FAISS, MongoDB Vector
- **Techniques:** RAG, PEFT, LoRA, PPO, Fine-tuning

### DevOps & CI/CD
- **CI/CD:** GitHub Actions, GitLab CI/CD, AWS CodePipeline
- **IaC:** AWS CDK, CloudFormation, Serverless Framework
- **Monitoring:** CloudWatch, X-Ray

### Integrations
- Stripe, Shopify, Calendly, Slack, Zoom, Google Calendar, Office365, RapidAPI

---

## Contact & Additional Information

For detailed project documentation, code samples, or consultation:
- **Portfolio Website:** geeksvisor.com
- Review project demos via the video links above
- Check UI/UX work on Figma

---

*Last Updated: March 2026*