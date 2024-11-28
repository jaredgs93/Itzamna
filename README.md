# SoftSkillsAssessment
A Multimodal Artificial Intelligence Platform for Comprehensive Soft Skills Assessment

This repository provides a system for evaluating soft skills using a multimodal approach. It includes a REST API and a Streamlit-based user interface for uploading videos, processing them, and generating detailed evaluation reports.

## Prerequisites

- **Docker** installed on your machine.
- An **OpenAI API Key** for accessing OpenAI services.
- A valid email address and password for sending reports.

---

## Setting Up the Environment

### Create the `.env` File

In the `src` folder, create a `.env` file with the following content:

```plaintext
OPENAI_API_KEY=your_openai_api_key
EMAIL_USER=your_email_address
EMAIL_PASSWORD=your_email_password
```
### How to Get Your OpenAI API Key

1. Go to the [OpenAI API Keys](https://platform.openai.com/account/api-keys) page.
2. Sign in or create an OpenAI account if you don’t already have one.
3. Click "Create new secret key" to generate a new API key.
4. Copy the API key and paste it into the `OPENAI_API_KEY` field in your `.env` file.

## Building and Running the Docker Image
