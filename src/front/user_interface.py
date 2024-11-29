import streamlit as st
import requests
import uuid
import tempfile
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


from decouple import config

# URL de la API
api_url_save_video = "http://127.0.0.1:8000/save-video"
api_url_evaluate = "http://127.0.0.1:8000/evaluate_skills"

st.set_page_config(layout="wide", page_title="Multimodal Assessment of Transversal Skills")

# Función para generar un nombre de archivo único
def generar_nombre_archivo():
    return f"{uuid.uuid4()}.mp4"

# Función para generar un archivo PDF con el contenido del informe
def generar_pdf(informe_texto):
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(temp_pdf.name, pagesize=letter)
    styles = getSampleStyleSheet()

    elementos = []
    elementos.append(Paragraph("Multimodal Assessment of Transversal Skills", styles["Heading1"]))
    elementos.append(Spacer(1, 12))

    for linea in informe_texto.split("\n"):
        linea = linea.strip()
        if linea.startswith("-"):
            elementos.append(Paragraph(linea, styles["Bullet"]))
        else:
            elementos.append(Paragraph(linea, styles["BodyText"]))
        elementos.append(Spacer(1, 6))

    doc.build(elementos)
    return temp_pdf.name

# Función para enviar correo electrónico con el informe adjunto
def enviar_correo(destinatario, asunto, cuerpo, archivo_adjunto):
    remitente = config("EMAIL_USER")
    password = config("EMAIL_PASSWORD")  # Usa una contraseña de aplicación si usas Gmail.

    # Configurar el mensaje
    mensaje = MIMEMultipart()
    mensaje["From"] = remitente
    mensaje["To"] = destinatario
    mensaje["Subject"] = asunto

    # Añadir el cuerpo del mensaje
    mensaje.attach(MIMEText(cuerpo, "plain"))

    # Adjuntar el archivo PDF
    with open(archivo_adjunto, "rb") as adjunto:
        parte = MIMEBase("application", "octet-stream")
        parte.set_payload(adjunto.read())
        encoders.encode_base64(parte)
        parte.add_header("Content-Disposition", f"attachment; filename={archivo_adjunto.split('/')[-1]}")
        mensaje.attach(parte)

    # Enviar el correo
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as servidor:
            servidor.starttls()
            servidor.login(remitente, password)
            servidor.sendmail(remitente, destinatario, mensaje.as_string())
            return True
    except Exception as e:
        st.error(f"Error sending email: {e}")
        return False

# Página principal para subir o grabar video
def page_upload():
    st.header("Multimodal Assessment of _Transversal Skills_")
    st.caption("Enter the URL of the video from the computer. The video must have a first-person shot of the speaker.")
    url_video_input = st.text_input("", "", key="url_video_input")  # Cambiar clave a url_video_input
    st.caption("Enter the topic for evaluation.")
    topic_input = st.text_input("", "", key="topic_input")  # Cambiar clave a topic_input
    send_video = st.button("Upload video URL", type="primary")
    if send_video:
        st.session_state["current_page"] = "processing"
        st.session_state["url_video"] = url_video_input  # Copiamos manualmente el valor
        st.session_state["task"] = "Skills"
        st.session_state["topic"] = topic_input  # Copiamos manualmente el valor
        st.session_state["evaluated"] = False  # Indicamos que aún no ha sido evaluado
        st.rerun()


# Página de procesamiento
def page_processing():
    if not st.session_state.get("evaluated", False):  # Solo evaluamos si no ha sido procesado
        with st.spinner("Evaluating video..."):
            topic = st.session_state.get("topic", "")
            url_video = st.session_state.get("url_video", "")
            data = {"video_url": url_video, "topic": topic}
            response = requests.post(api_url_evaluate, json=data)

            if response.status_code == 200:
                salida_api = response.json()
                st.session_state["report"] = salida_api["report"]
                st.session_state["details"] = salida_api["details"]
                st.session_state["person_id"] = salida_api["id"]
                st.session_state["processing_time"] = round(
                    float(response.headers["x-process-time"].replace(" sec", "")), 1
                )
                st.session_state["evaluated"] = True
            else:
                st.error("Error processing video")
                return

    # Mostrar resultados y detalles
    st.header("Multimodal Assessment of _Transversal Skills_")
    st.success(
        f"Results available! Processing time: {st.session_state['processing_time']} seconds."
    )

    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        st.video(st.session_state["url_video"], autoplay=True, loop=True)
    with col2:
        st.subheader("📈 Results")
        for metrica in st.session_state["details"]:
            nombre = metrica.replace("_", " ").capitalize()
            col2.progress(
                st.session_state["details"][metrica]["score"] / 10,
                text=f"{nombre}: {st.session_state['details'][metrica]['label']}",
            )

    st.subheader("📁 Detailed report")
    st.write(st.session_state["report"])

    pdf_path = generar_pdf(st.session_state["report"])
    with open(pdf_path, "rb") as pdf_file:
        st.download_button(
            label="📄 Download report as PDF",
            data=pdf_file,
            file_name=f"{st.session_state['person_id']}_transversal_skills_report.pdf",
            mime="application/pdf",
        )

    # Formulario para enviar el informe por correo electrónico
    st.subheader("📧 Send report via email")
    email_input = st.text_input("Enter recipient email:")
    if st.button("Send Email"):
        if email_input:
            if enviar_correo(
                email_input,
                "Transversal Skills Assessment Report",
                "Please find attached the Transversal Skills Assessment report.",
                pdf_path,
            ):
                st.success("Email sent successfully!")
        else:
            st.error("Please enter a valid email address.")

    if st.button("Back to Home"):
        st.session_state["current_page"] = "upload"
        st.session_state["evaluated"] = False  # Reiniciamos el estado para una nueva evaluación
        st.rerun()

# Main
def main():
    st.session_state.setdefault("current_page", "upload")
    st.session_state.setdefault("evaluated", False)  # Inicializamos el estado "evaluated"

    if st.session_state["current_page"] == "upload":
        page_upload()
    elif st.session_state["current_page"] == "processing":
        page_processing()

if __name__ == "__main__":
    main()
