# Utilisation d'une image Python officielle en tant que base
FROM python:3.9

# Copie des fichiers de votre dépôt dans le conteneur
WORKDIR /app
COPY . /app

# Installation des dépendances nécessaires (si vous avez un fichier requirements.txt)
RUN pip install -r requirements.txt

# Commande pour exécuter main.py
CMD ["python", "main.py"]