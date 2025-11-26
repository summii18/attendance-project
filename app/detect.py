from facenet_pytorch import MTCNN
from PIL import Image
import torch
from models.load_models import get_models

# Load all models once
embedding_model, classifier_model, label_encoder, device = get_models()
mtcnn = MTCNN(keep_all=False, device=device)

def recognize_student(pil_image):
    face = mtcnn(pil_image)

    if face is None:
        return None, 0

    face = face.unsqueeze(0).to(device)

    with torch.no_grad():
        emb = embedding_model(face)
        logits = classifier_model(emb)
        prob = torch.softmax(logits, dim=1)
        top_prob, top_index = prob.max(dim=1)

    if top_prob < 0.60:
        return "Unknown", float(top_prob)

    student_id = label_encoder.inverse_transform([top_index.item()])[0]
    return student_id, float(top_prob)