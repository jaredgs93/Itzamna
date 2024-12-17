# Transversal Skills Assessment
A Multimodal Artificial Intelligence Platform for Comprehensive Transversal Skills Assessment

This repository provides a system for evaluating skills using a multimodal approach. It includes a REST API and two Streamlit-based user interfaces:
1. **User Interface for Video Evaluation**: Allows users to upload videos, process them, and generate detailed evaluation reports. These reports can be downloaded as PDFs or sent via email.
2. **Rules Creation Interface**: Enables users to create and manage evaluation rules, defining the antecedents and consequents for skill assessment.

![overview](https://github.com/user-attachments/assets/2a532cd1-4a4c-4237-ba7c-39bbe4cf3603)


## Table of Contents

1. [Project Structure](#project-structure)
2. [Prerequisites](#prerequisites)
3. [System Composition with Docker](#system-composition-with-docker)
4. [Building and Running the Docker Image](#building-and-running-the-docker-image)
5. [How to Use](#how-to-use)
---
## Project Structure
The repository is organized as follows:
```bash
src/
├── app/                  # Contains Python scripts and modules for the API
│   ├── __init__.py       # Initializes the app module
│   ├── audio.py          # Handles audio processing for video evaluation
│   ├── inference.py      # Contains machine learning inference logic
│   ├── main.py           # Entry point for the API
│   ├── myprosody2.py     # Processes prosodic features for speech analysis
│   ├── processing.py     # Handles data processing and integration tasks
│   ├── report_generation.py # Generates detailed reports based on evaluations
│   ├── text.py           # Processes textual data extracted from videos
│   ├── translation.py    # Handles translation tasks, if applicable
│   └── video.py          # Manages video-related operations
│   └── ...               # Additional scripts and modules
├── data/                 # Contains data files and MongoDB initialization script
│   ├── fuzzy_variables.json   # JSON file for initializing fuzzy variables
│   ├── rules_evaluation.json  # JSON file for initializing rules
│   └── init-mongo.sh     # Script to load data into MongoDB
├── front/                # Contains Streamlit-based user interfaces
│   ├── rules_creation.py # Interface for creating evaluation rules
│   ├── user_interface.py # Interface for uploading videos and viewing reports
├── .gitattributes        # Configuration for Git LFS
├── .gitignore            # Files and directories ignored by Git
├── Dockerfile            # Instructions to build the Docker image
├── LICENSE               # License file for the repository
├── README.md             # Documentation for the repository
├── docker-compose.yml    # Orchestration of the services with Docker
├── environment-streamlit.yml # Conda environment configuration for Streamlit
└── environment.yml       # Conda environment configuration for the API
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
git clone
cd TransversalSkillsAssessment
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
- Enter the video URL you want to evaluate. The URL must point directly to a video file and end with the file extension (e.g., `.mp4`, `.mov`, etc.).
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
  - Specify the resulting skill and its value (e.g., Leadership Security is Low).

#### 3. Save the Rule
- Click the "Save Rule" button to store the new rule.
- A confirmation message, "Rule saved successfully!", will appear upon successful saving.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



