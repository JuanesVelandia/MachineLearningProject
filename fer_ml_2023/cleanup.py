import os
import shutil

def cleanup():
    media_folder = 'media/'  # Ruta de la carpeta de medios donde se guardan las imágenes

    if os.path.exists(media_folder):
        shutil.rmtree(media_folder)
        os.makedirs(media_folder)
    else:
        os.makedirs(media_folder)

if __name__ == "__main__":
    cleanup()
