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
### 1. Build the Docker Image
To build the Docker image, run the following command in the terminal from the root directory of this repository:
```bash
docker build -t videoevaluation .
```

### 2. Create and Run the Docker Container
```bash
docker run -p 8000:8000 -p 8501:8501 -p 8502:8502 --env-file src/.env videoevaluation
```
## How to Use
### Video evaluation
#### 1. Upload a Video
- Access the Streamlit interface at http://localhost:8501.
- Upload a video URL and provide the topic for evaluation.

#### 2. Process the Video
The backend API processes the video, evaluates soft skills, and generates a detailed report.

#### 3. Download or Email the Report
- Download the report as a PDF.
- Optionally, send the report to an email address.

### Rules creation
#### 1. Access the Rules Creation Interface
- Open the rules creation interface at [http://localhost:8502](http://localhost:8502).

#### 2. Create a New Rule
- Fill in the **Antecedents**:
  - Select the desired conditions from the dropdowns for each antecedent (e.g., Gaze, Voice Uniformity, Serenity).
  - Specify if the condition **IS** or **IS NOT** met.
  - Enter a value for the condition (e.g., Centered, Low).
  - Choose an **Operator** (e.g., AND, OR) to combine antecedents.

- Define the **Consequent**:
  - Specify the resulting soft skill and its value (e.g., Leadership Security is Low).

#### 3. Save the Rule
- Click the "Save Rule" button to store the new rule.
- A confirmation message, "Rule saved successfully!", will appear upon successful saving.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



