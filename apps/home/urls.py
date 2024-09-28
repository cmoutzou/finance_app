# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.urls import path, re_path
from apps.home import views

urlpatterns = [

    # The home page
    path('', views.index, name='home'),
    path('get_chartBig1_data/', views.get_chartBig1_data, name='get_chartBig1_data'),
    path('get_chartpurple_data/', views.get_chartpurple_data, name='get_chartpurple_data'),
    path('get_CountryChart_data/', views.get_CountryChart_data, name='get_CountryChart_data'),
    path('create_charts/', views.create_charts, name='create_charts'),
    path('get_ticker/', views.get_ticker, name='get_ticker'),
    # Matches any html file
    re_path(r'^.*\.*', views.pages, name='pages'),

]
