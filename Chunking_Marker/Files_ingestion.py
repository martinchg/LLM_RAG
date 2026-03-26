import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
# from office365.sharepoint.client_context import ClientContext
# from office365.runtime.auth.user_credential import UserCredential

def collecte_fichiers_locaux(file_path):
    fichiers = []
    extensions = [".pdf", ".docx", ".txt", ".html", ".mp3", ".wav",".xlsx", ".xls"]
    
    for f in Path(file_path).rglob("*"):
        if f.suffix.lower() in extensions:
            fichiers.append(f)
    
    return fichiers

def collecte_sharepoint_fichiers(site_url, folder_url, username, password):
    extensions = [".pdf", ".docx", ".txt", ".html"]
    fichiers=[]
    # collecte des fichiers dans le dossier et les sous dossiers d'une façon récursive
    def collect_sharepoint_files(folder):
        ctx.load(folder.files)
        ctx.load(folder.folders)
        ctx.execute_query()

        # Ajouter les fichiers du dossier courant
        for f in folder.files:
            if Path(f.properties["Name"]).suffix.lower() in extensions:
                fichiers.append(f)
        
        # Parcourir récursivement les sous-dossiers
        for subfolder in folder.folders:
            collect_sharepoint_files(subfolder)

    ctx = ClientContext(site_url).with_credentials(UserCredential(username, password))
    root_folder = ctx.web.get_folder_by_server_relative_url(folder_url)
    collect_sharepoint_files(root_folder)
    
    return fichiers

def copy_local_files(fichiers, destination_folder):
    Path(destination_folder).mkdir(parents=True, exist_ok=True)
    for f in fichiers:
        dest_path = Path(destination_folder) / f.name
        with open(f, "rb") as src, open(dest_path, "wb") as dst:
            dst.write(src.read())

def download_sharepoint_files(fichiers, destination_folder):
    Path(destination_folder).mkdir(parents=True, exist_ok=True)
    for f in fichiers:
        local_path = Path(destination_folder) / f.properties["Name"]
        with open(local_path, "wb") as local_file:
            f.download(local_file).execute_query()

