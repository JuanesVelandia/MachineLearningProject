from django.shortcuts import render
from .models import FacialExpressionModel
import os, base64
from django.conf import settings
from django.core.files.base import ContentFile


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'myapp/MLmodels/fer2013.pt')


def main(request):
    return render(request, 'main.html')


def processing(request):
    if request.method == 'POST':
        if 'imagen_usuario' in request.FILES:
            image_file = request.FILES['imagen_usuario']
            print(image_file)
        else:
            imgstr = request.POST.get('captured_image')

            format, imgstr = imgstr.split(';base64,')
            ext = format.split('/')[-1]

            data = ContentFile(base64.b64decode(imgstr), name=f'picture.{ext}')
            image_file = data
        
        with open(os.path.join(settings.MEDIA_ROOT, image_file.name), 'wb') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)


        model = FacialExpressionModel(model_path=model_path)
        resultado = model.predict(image_file)
        return render(request, 'main.html', {'prediction': resultado})
