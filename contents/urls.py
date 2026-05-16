from django.urls import path
from .views import (
    ExploreView,
    BoardGameListView, BoardGameDetailView,
    EscapeListView, EscapeDetailView,
    CrimeSceneListView, CrimeSceneDetailView,
)
 
app_name = "contents"
 
urlpatterns = [
    path("explore/",             ExploreView.as_view(),          name="explore"),
    path("boardgame/",           BoardGameListView.as_view(),    name="boardgame_list"),
    path("boardgame/<int:pk>/",  BoardGameDetailView.as_view(),  name="boardgame_detail"),
    path("escape/",              EscapeListView.as_view(),       name="escape_list"),
    path("escape/<int:pk>/",     EscapeDetailView.as_view(),     name="escape_detail"),
    path("crimescene/",          CrimeSceneListView.as_view(),   name="crimescene_list"),
    path("crimescene/<int:pk>/", CrimeSceneDetailView.as_view(), name="crimescene_detail"),
]