import os
import base64
from django.conf import settings
from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from .serializers import FileSerializer
from prediction2 import predict


class FileUploadView(APIView):
    parser_class = (FileUploadParser,)

    def post(self, request, *args, **kwargs):
      file_serializer = FileSerializer(data=request.data)
      if file_serializer.is_valid():
          try:
              res = predict(file_serializer.validated_data["file"],file_serializer.validated_data["file2"])
              with open(os.path.join(settings.BASE_DIR, 'change.png'), "rb") as image_file:
                  image_data = base64.b64encode(image_file.read()).decode('utf-8')
              res = {"result":image_data}
              return Response(res, status=status.HTTP_201_CREATED)
          except Exception as e:
              predictions = {"error": "2", "message": str(e)}
              return Response(predictions, status=status.HTTP_201_CREATED)
      else:
          return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
