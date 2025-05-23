# Itzamna
A Multimodal Artificial Intelligence Platform for Comprehensive Transversal Skills Evaluation

This repository provides a system for evaluating skills using a multimodal approach. It includes a REST API and two Streamlit-based user interfaces for evaluating decision-making, negotiation, leadership, stress control, creativity and self-steem:
1. **User Interface for Video Evaluation**: Allows users to upload videos, process them, and generate detailed evaluation reports. These reports can be downloaded as PDFs or sent via email.
2. **Rules Creation Interface**: Enables users to create and manage evaluation rules, defining the antecedents and consequents for skill assessment.

![overview](https://github.com/user-attachments/assets/2a532cd1-4a4c-4237-ba7c-39bbe4cf3603)

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [How Skills Are Evaluated](#how-skills-are-evaluated)
3. [Prerequisites](#prerequisites)
4. [System Composition with Docker](#system-composition-with-docker)
5. [Building and Running the Docker Image](#building-and-running-the-docker-image)
6. [How to Use](#how-to-use)
7. [Database Structure](#database-structure)
8. [API Endpoints](#api-endpoints)
9. [Example Execution](#example-execution)
10. [Performance and Resource Requirements](#performance-and-resource-requirements)
---
## Project Structure
The repository is organized as follows:
```bash
examples/
├── sample_outputs/       # Contains all output files generated after evaluating the example video
│   ├── audio.wav              # Extracted audio track from the video
│   ├── blinking.json          # Detected blinks per frame
│   ├── emotions.json          # Detected emotional expressions across the video
│   ├── faces_smiles.json      # Facial landmarks and smile detection per frame
│   ├── fuzzy_control_output.json # Final fuzzy evaluation results and linguistic labels
│   ├── measures_for_inference.json # Aggregated metrics used as input to the fuzzy system
│   ├── metadata.json          # Video metadata (e.g., duration, resolution)
│   ├── prosody.TextGrid       # Prosodic features: pitch, rhythm, timing
│   ├── skills_explanation.txt # Human-readable breakdown of assessed skills and subcomponents
│   ├── text_measures.json     # Linguistic analysis: vocabulary richness, structure, etc.
│   ├── topics.json            # Key topics detected in the spoken content
│   └── transcription.json     # Full transcription of the spoken audio
├── video_example.mp4          # Sample video used to illustrate the system's full pipeline
src/
├── app/                       # Contains Python scripts and modules for the API
│   ├── __init__.py            # Initializes the app module
│   ├── audio.py               # Handles audio processing for video evaluation
│   ├── inference.py           # Contains machine learning inference logic
│   ├── main.py                # Entry point for the API
│   ├── myprosody2.py          # Processes prosodic features for speech analysis
│   ├── processing.py          # Handles data processing and integration tasks
│   ├── report_generation.py   # Generates detailed reports based on evaluations
│   ├── text.py                # Processes textual data extracted from videos
│   ├── translation.py         # Handles translation tasks, if applicable
│   └── video.py               # Manages video-related operations
│   └── ...                    # Additional scripts and modules
├── data/                      # Contains data files and MongoDB initialization script
│   ├── fuzzy_variables.json   # JSON file for initializing fuzzy variables
│   ├── rules_evaluation.json  # JSON file for initializing rules
│   └── init-mongo.sh          # Script to load data into MongoDB
├── front/                     # Contains Streamlit-based user interfaces
│   ├── rules_creation.py      # Interface for creating evaluation rules
│   ├── user_interface.py      # Interface for uploading videos and viewing reports
├── .gitattributes             # Configuration for Git LFS
├── .gitignore                 # Files and directories ignored by Git
├── Dockerfile                 # Instructions to build the Docker image
├── LICENSE                    # License file for the repository
├── README.md                  # Documentation for the repository
├── docker-compose.yml         # Orchestration of the services with Docker
├── environment-streamlit.yml  # Conda environment configuration for Streamlit
└── environment.yml            # Conda environment configuration for the API
```

### Details
   - `app/`: Contains the core logic and modules for the API, including its main entry point (`main.py`).
   - `data/`:
      - `fuzzy_variables.json`: Predefined fuzzy variables used by the system.
      - `rules_evaluation.json`: Predefined evaluation rules.
      - `init-mongo.sh`: A script that initializes MongoDB with the provided JSON files.
   - `front/`:
      - `rules_creation.py`: A Streamlit interface for creating and managing evaluation rules.
      - `user_interface.py`: A Streamlit interface for uploading videos, processing them, and generating evaluation reports.

---
## How Skills Are Evaluated
Each transversal skill in Itzamna is evaluated using a Granular Linguistic Model of Phenomena (GLMP), a fuzzy logic-based framework designed for interpretable and layered assessment. This model organises the evaluation process hierarchically across four levels:
1. Transversal Skill: The high-level competency being assessed.
2. Dimensions: Core aspects that define the skill and reflect broader behavioural tendencies.
3. Attributes: Specific behavioural expressions or tendencies that contribute to each dimension.
4. Measures: Quantitative indicators extracted from audio, text, or video input

### Measures Catalogue by Modality
The evaluation of transversal skills in Itzamna relies on behavioural measures extracted from three main modalities. These measures are mapped to fuzzy labels and contribute to higher-level dimensions and skills.

This system uses 26 behavioural measures grouped by modality. Each measure is quantified and normalised into a predefined scale.

| **Modality** | **Measure**           | **Description** | **Scale** |
|--------------|-----------------------|------------------|-----------|
| **Audio**    | Consistency           | Absence of long pauses or hesitations in speech. Ratio of pause time over total time. | [0,10] |
|              | Fluency               | Number of pauses and interruptions per minute. | [0,50] |
|              | Mood                  | Vocal tone detection: reading, neutral, passionate. | [0,4] |
|              | Noise                 | Detection of background or external sounds (e.g., throat clearing). Ratio of noise presence. | [0,10] |
|              | Reaction time         | Time taken to begin speaking after a stimulus, in seconds. | [0,4] |
|              | Speech speed          | Number of syllables per second. | [0,8] |
|              | Voice uniformity      | Corrected for average pitch and variation, complementing mood detection. | [0,10] |
| **Text**     | Concreteness          | Number of argumentative, reinforcing or concrete linguistic markers per minute relative to topics. | [0,10] |
|              | Congruence            | Semantic consistency across topics. Avoids unrelated or mixed subjects. | [0,10] |
|              | Crutches              | Use of filler words in the text per minute. | [0,10] |
|              | Density               | Number of topics discussed, scaled logarithmically. | [0,10] |
|              | Examples              | Number of adjectives/examples accompanying the explanation per minute. | [0,50] |
|              | Order                 | Detection and sequencing of discourse markers (e.g., beginning, body, conclusion). | [0,10] |
|              | Organization          | Use of discourse connectors (normalised per minute). | [0,10] |
|              | Originality           | Use of uncommon or topic-specific vocabulary. Ratio of low-frequency words. | [0,10] |
|              | Quantity              | Number of relevant topics addressed. | [0,10] |
|              | Redundancy            | Frequency of repeated content (excluding stop words). | [0,10] |
|              | Respect               | Appropriate, polite, and socially acceptable language. Includes greetings/farewells. | [0,10] |
|              | Topic adequacy        | Relevance and thematic alignment between the speech and the original questions. | [0,10] |
|              | Vagueness             | Use of ambiguous or imprecise expressions. | [0,10] |
|              | Verbal tense          | Percentage of verbs used in the present tense over total verbs. | [0,10] |
| **Video**    | Blinking              | Frequency of blinking per minute, as a stress indicator. (Capped at 10). | [0,10] |
|              | Gaze                  | Ratio of focused gazes over total gaze movements. | [0,10] |
|              | Gesture               | Balance between positive, negative, and stress gestures. Computed as (10 - ratio of neg/pos expressions). | [0,10] |
|              | Posture changes       | Number of head or body shifts per minute (horizontal, vertical, focal). | [0,10] |
|              | Smile                 | Proportion of frames where a smile is detected. | [0,10] |

Each of these measures is automatically transformed into **linguistic labels** (e.g., *Low*, *Medium*, *High*) based on its defined scale. These labels are then processed through fuzzy logic rules to compute **attributes**, which are intermediate behavioural constructs.

Subsequently, **attributes are aggregated to form dimensions**, and **dimensions are combined to generate the final evaluation for each transversal skill**.

While the original measures may use different scales, **attributes, dimensions, and skills are all normalised to a [0,10] scale**, ensuring consistency and interpretability across the entire assessment framework.


### Evaluated Transversal Skills
Itzamna currently assesses six transversal skills using structured fuzzy models. Each skill is represented as a hierarchical diagram detailing its structure for evaluation. These models ensure transparency and interpretability, and can be adapted or extended based on specific application needs.

#### 1. Decision-making 
Focuses on the user's ability to express ideas clearly, maintain conciseness, and convey firmness in their responses. 
![skill_1-decision_making](https://github.com/user-attachments/assets/481fd0ed-e680-4ced-98f0-4fc965692f1e)

#### 2. Negotiation
Assesses argumentation quality, expression clarity, and non-verbal empathy during interactive discourse.
![skill_2-negotiation](https://github.com/user-attachments/assets/96975414-6d71-42d1-80f2-d56c3482b346)

#### 3. Leadership
Evaluates confidence, persuasive clarity, and coherence in delivery, including non-verbal assertiveness.
![skill_3-leadership](https://github.com/user-attachments/assets/87d13c7f-091c-4f85-aae1-0a1a79bbb3ad)

#### 4. Stress Control
Monitors nervousness, consistency in communication, and control of body language under pressure.
![skill_4-stress_control](https://github.com/user-attachments/assets/d0c61ccd-b04b-4eac-a452-0544a69a9e3d)

#### 5. Creativity
Measures originality, lexical richness, idea generation, and expressive variety across modalities.
![skill_5-creativity](https://github.com/user-attachments/assets/6b7ac41b-0a43-4952-9098-47d871e5a016)

#### 6. Self-esteem
Analyses indicators of self-confidence and emotional tone through posture, fluency, and expressive content.
![skill_6-self_esteem](https://github.com/user-attachments/assets/c4fd613f-1c25-4a6b-85f1-a70cc0703f7d)

---
## Prerequisites

- **Docker** installed on your machine.
- An **OpenAI API Key** for accessing OpenAI services.
- A **Gmail account** with an App Password enabled. This is required for sending reports via email, as the system is currently designed to use Gmail's SMTP server.

### How to Get Your OpenAI API Key

1. Go to the [OpenAI API Keys](https://platform.openai.com/account/api-keys) page.
2. Sign in or create an OpenAI account if you don’t already have one.
3. Click "Create new secret key" to generate a new API key.
4. Copy the API key and paste it into the `OPENAI_API_KEY` field in your `.env` file.

### Steps to Create an App Password for Gmail

1. Enable Two-Step Verification
   - Go to your [Google Account](https://myaccount.google.com/).
   - Click on **Security** in the left menu.
   - Under the **How you sign in to Google**, enable **2-Step Verification** if it’s not already activated.
   - Follow the instructions to set up 2-Step Verification (you can use text messages, apps like Google Authenticator, or physical security keys).

2. Access the App Passwords Section
   - Once **2-Step Verification** is enabled, [login to create and manage application passwords](https://myaccount.google.com/apppasswords).
   - In the **To create a new app-specific password, type a name for it below…** enter a descriptive name like `DockerApp`.
   - Click **Create**.

3. Copy the Generated Password
   - A 16-character password will appear in a pop-up window.
   - Copy this password. It will only be shown once, so make sure to save it securely.
---

## System Composition with Docker
This system uses Docker Compose to orchestrate multiple services, ensuring seamless integration and ease of deployment. Below is an overview of the services included:

### Services
#### 1. API Service (`api`)
   - Description: This service hosts the REST API for skill evaluation, which processes videos, creates rules, and handles requests from the user interfaces.
   - Exposed Port: `8000`
   - Build: Built using the `Dockerfile` in the root directory.
   - Volume: Maps the local `src` directory to `/app/src` in the container for code synchronization.
   - Dependencies: Depends on the `mongodb` service for database operations.

#### 2. Streamlit Rule Creation Interface (`streamlit`)
   - Description: A Streamlit-based interface for creating and managing evaluation rules.
   - Exposed Port: `8501`
   - Command: Runs the `rules_creation.py` interface located in `src/front`.

#### 3. Streamlit User Evaluation Interface (`streamlit-user-interface`)
   - Description: A Streamlit-based interface for uploading videos, evaluating them, and generating detailed reports.
   - Exposed Port: `8502`
   - Command: Runs the `user_interface.py` interface located in `src/front`.

#### 4. MongoDB Service (`mongodb`)
   - Description: A MongoDB database used for storing rules, evaluation data, and other related information.
   - Exposed Port: `27017`
   - Data Persistence: Uses a Docker volume (`mongo-data`) to persist database files.
   - Initialization: Initializes with data from the `src/data` directory.

### Volumes
   - `mongo-data`: A Docker volume used to persist MongoDB data across container restarts.
---

## Building and Running the Docker Image
### 1. Clone the Repository
Start by cloning the repository and pulling the required files (including those managed by Git LFS):
```bash
git clone https://github.com/jaredgs93/Itzamna.git
cd Itzamna
git lfs install
git lfs pull
```

### 2. Create the `.env` File

In the `src` folder, create a `.env` file with the following content:

```plaintext
OPENAI_API_KEY=your_openai_api_key
EMAIL_USER=your_gmail_address
EMAIL_PASSWORD=your_app_password
```

### 3. Build and Run the Services
Use `docker-compose` to build and run the services. This command will build the Docker images and start all necessary containers:
```bash
docker-compose up --build
```
### 4. Access the Services
Once the containers are running, you can access the following services:
- **API**: `http://<SERVER_IP>:8000`
- **Rules Creation Interface**: `http://<SERVER_IP>:8501`
- **User Interface**: `http://<SERVER_IP>:8502`

Replace `<SERVER_IP>` with:

- `localhost` if running locally.
- The public IP address or domain name of your server if running remotely.

### 5. Stop the Services
To stop the containers and free resources, use:
```
docker-compose down
```
---
## How to Use
### Video evaluation
#### 1. Access the User Interface
- Access the interface at `http://<SERVER_IP>:8502`, replacing `<SERVER_IP>` with:
  - `localhost` if running locally.
  - The public IP address or domain name of the server if running remotely (e.g., `http://123.45.67.89:8502`).
#### 2. Upload a Video
- Enter the video URL you want to evaluate. The URL must point directly to a video file and end with the file extension (e.g., `.mp4`, `.mov`, etc.). **The minimum dimensions of the video must be 1280 x 720 and the duration must be between 30 and 210 seconds. The video should include close-up shots of a person making a speech, which, for best results, should be in Spanish or English.**
- Provide the topic for evaluation in the text input field.
- Click **Upload Video URL** to start the evaluation process.

<img width="1440" alt="home_page" src="https://github.com/user-attachments/assets/442e272b-ddd0-4815-9374-46b830d71341" />

#### 3. Process the Video

The backend API processes the video, evaluates skills, and generates a detailed report. Once the procedure is finished, at the top of the page you can view the assessed video and a summary of the results of the transversal skills.

![results_interface_1](https://github.com/user-attachments/assets/1f9a0e93-f963-49ee-a312-933810df310d)

#### 4. Download or Email the Report
Below that, you can view the detailed report of the results. You can also:
- Download the report as a PDF.
- Optionally, send the report to an email address.

<img width="1313" alt="results_interface_2" src="https://github.com/user-attachments/assets/2bd5b587-e246-45cf-93d0-a59964967ad5" />


### Rules creation
![rules_interface](https://github.com/user-attachments/assets/6a4e02b9-beec-41df-a03c-18c7df9b6786)

#### 1. Access the User Interface
- Access the interface at `http://<SERVER_IP>:8501`, replacing `<SERVER_IP>` with:
  - `localhost` if running locally.
  - The public IP address or domain name of the server if running remotely (e.g., `http://123.45.67.89:8501`).

#### 2. Create a New Rule
- Fill in the **Antecedents**:
  - Select the desired conditions from the dropdowns for each antecedent (e.g., Gaze, Voice Uniformity, Serenity).
  - Specify if the condition **IS** or **IS NOT** met.
  - Enter a value for the condition (e.g., Centered, Low).
  - Choose an **Operator** (e.g., AND, OR) to combine antecedents.

- Define the **Consequent**:
  - Specify the resulting consequent and its value (e.g., Leadership Security is Low).

#### 3. Save the Rule
- Click the "Save Rule" button to store the new rule.
- A confirmation message, "Rule saved successfully!", will appear upon successful saving.

---

## Database Structure

Itzamna uses a MongoDB database named `skills_evaluation` that contains the necessary knowledge base for the fuzzy inference system. It comprises two collections: `fuzzy_variables` and `rules_evaluation`.


#### Collection: `fuzzy_variables`

This collection defines all fuzzy variables used in the evaluation system, including both behavioural measures and intermediate constructs (attributes or dimensions).

| Field                  | Type       | Description                                                                 |
|------------------------|------------|-----------------------------------------------------------------------------|
| `_id`                 | `ObjectId` | Unique identifier automatically generated by MongoDB.                      |
| `variable_name`       | `String`   | Internal name of the variable, usually in Spanish  |
| `variable_description`| `String`   | Human-readable description of the variable in Spanish.                     |
| `variable_description_en` | `String` | Human-readable description of the variable in English.                     |
| `fuzzy_sets`          | `Array`    | List of fuzzy labels (Spanish labels).      |
| `fuzzy_sets_en`       | `Array`    | List of fuzzy labels (English labels).      |
| `measure`             | `Boolean`  | Present only if the variable is a behavioural measure extracted from input. |


#### Collection: `rules_evaluation`

This collection contains the fuzzy inference rules used to evaluate attributes, dimensions, and skills.

| Field              | Type       | Description                                                                 |
|--------------------|------------|-----------------------------------------------------------------------------|
| `_id`             | `ObjectId` | Unique identifier automatically generated by MongoDB.                      |
| `rule_id`         | `Integer`  | Numeric ID used to identify and order rules.                               |
| `antecedent`      | `String`   | Logical expression using fuzzy labels and variables.                       |
| `consequent`      | `String`   | Name of the output variable the rule refers to (usually an attribute, dimension, or skill). |
| `consequent_value`| `String`   | Linguistic label assigned to the consequent if the rule is satisfied.      |

These rules follow a human-readable structure (e.g., `IF X[Low] AND Y[High]`) and are interpreted by the fuzzy inference engine.

---
## API Endpoints
Below is a list of available API endpoints with their descriptions, example requests, and expected input:

### 1. GET /antecedents
Retrieves a list of potential antecedents (input variables) from the fuzzy variables collection, which users can employ to define conditions for fuzzy rules. Each antecedent is accompanied by detailed descriptions and fuzzy sets.

**Example Request**
```bash
curl -X 'GET' \
  'http://<SERVER_IP>:8000/antecedents' \
  -H 'accept: application/json'
```

### 2. GET /consequents
Returns a list of consequents from the fuzzy variables collection, representing potential outcomes for the fuzzy rules. Each consequent includes descriptive information and associated fuzzy sets, which define the possible levels of each consequent.

**Example Request**
```bash
curl -X 'GET' \
  'http://<SERVER_IP>:8000/consequents' \
  -H 'accept: application/json'
```

### 3. POST /create_rule
Permits users to define new fuzzy rules by indicating antecedents, consequents, and consequent values. These rules are stored in a MongoDB collection, thereby facilitating the creation of bespoke configurations for skills assessments aligned with the organisational context's specific requirements.

**Request Body**
```bash
{
  "antecedent": "string",
  "consequent": "string",
  "consequent_value": "string"
}
```
**Example Request**
```bash
curl -X 'POST' \
  'http://<SERVER_IP>:8000/create_rule' \
  -H 'Content-Type: application/json' \
  -d '{
    "antecedent": "IF organization [High] AND mood[Passionately]",
    "consequent": "firmness",
    "consequent_value": "High"
}'
```
### 4. POST /check_video_eligibility
Verifies whether a video adheres to the specified criteria (e.g., duration, resolution) for evaluation. If the video fails to meet the above mentioned criteria, the endpoint will return a detailed message explaining the ineligibility.

**Request Body**
```bash
{
  "video_url": "string",
  "topic": "string"
}
```
**Example Request**
```
curl -X 'POST' \
  'http://<SERVER_IP>:8000/check_video_eligibility' \
  -H 'Content-Type: application/json' \
  -d '{
    "video_url": "http://<SERVER_IP>/video/speech.mp4",
    "topic": "Research in computer science"
}'
```

### 5. POST /evaluate_skills

Initiates a comprehensive assessment of an eligible video. This endpoint accepts two parameters: the URL of the video to be evaluated and the topic the user is expected to discuss. Initially, it verifies the video's eligibility through the \textit{/check\_video\_eligibility} endpoint. Subsequently, it conducts multimedia analyses through different AI techniques. The outcomes of these analyses are then synthesised into a structured skills assessment.

**Request Body**
```bash
{
  "video_url": "string",
  "topic": "string"
}
```
**Example Request**
```
curl -X 'POST' \
  'http://<SERVER_IP>:8000/evaluate_skills' \
  -H 'Content-Type: application/json' \
  -d '{
    "video_url": "http://<SERVER_IP>/video/speech.mp4",
    "topic": "Research in computer science"
}'
```
---

## Example Execution

To facilitate replication and demonstrate the output structure, the repository includes a working example located in the `examples/` directory.

- `video_example.mp4`: A sample evaluation video, already anonymised.
- `sample_outputs/`: The output files automatically generated by the system after processing the video.

Each output file corresponds to a specific analysis stage, such as:
- Video metadata: `metadata.json`
- Extracted audio: `audio.wav`
- Transcription: `transcription.json`
- Facial and emotional analysis: `faces_smiles.json`, `emotions.json`, `blinking.json`
- Text and prosodic features: `text_measures.json`, `prosody.TextGrid`, `topics.json`
- Fuzzy system inputs and outputs: `measures_for_inference.json`, `fuzzy_control_output.json`
- Final report input and output: `skills_explanation.txt`

### How to Run the Example

1. Access the User Interface
2. In the "Multimodal Assessment of Transversal Skills" form:
   - Enter the following video path:
   ```bash
   /absolute/path/to/Itzamna/examples/video_example.mp4
   ```
3. Set the topic to:
   ```bash
   Computer science
   ```
4. Click on `Upload video URL` to start the evaluation.

This example showcases the complete evaluation pipeline and enables users to inspect both the frontend summary and the raw structured outputs.

---
## Performance and Resource Requirements

To validate the system’s scalability and suitability for real-world deployment, a series of performance benchmarks were conducted using a Linode dedicated server with the following specifications:

- **RAM:** 8 GB  
- **CPU:** 4 virtual cores  
- **Disk:** 160 GB SSD  
- **Bandwidth:** 40 Gbps (in) / 5 Gbps (out)  

### Resource Usage

The system loads all necessary components—such as Whisper, DeepFace, and transformer-based language models—at runtime. Initialisation requires approximately **5.94 GB of memory**, with **peak memory usage staying below 8 GB** during evaluation. Memory is released upon completion of each task. CPU usage peaks at **65% during system loading**, and remains below **25% while processing** a video.

### Processing Time

Input videos ranged between **1.5 and 3 minutes** in length. The **average evaluation time per video** (from data loading to final report generation) was approximately **72 seconds** under sequential execution.

### Disk Usage

The system requires approximately **12.9 GB** of disk space for model files and dependencies. Each video evaluation temporarily uses **15–18 MB** of disk space for intermediate processing files. Videos uploaded in unsupported formats are automatically converted to `.mp4`, adding minimal extra disk usage.

### Memory Usage Snapshot

| Evaluation Step      | Memory Usage (GB) |
|----------------------|-------------------|
| System Initialisation| 5.94              |
| Evaluation #1        | 6.72              |
| Evaluation #2        | 6.65              |
| Evaluation #3        | 6.60              |
| Evaluation #4        | 6.74              |
| Evaluation #5        | 6.72              |
| Final Memory State   | 5.92              |

> All memory values were obtained using the `psutil` library on Linux under Python 3.10.
---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



