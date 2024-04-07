from django.contrib import admin
from django.urls import path
from LvqPso.views import *

urlpatterns = [
    path("", App.index, name="index"),
    path("admin/", admin.site.urls),
]
