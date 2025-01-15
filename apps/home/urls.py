# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.urls import path, re_path
from . import views

urlpatterns = [

    # The home page
    path('', views.index, name='home'),
    path('portfolio_view/', views.portfolio_view, name='portfolio_view'),
    path('portfolio_data/', views.portfolio_data, name='portfolio_data'),
    path('asset_category_performance_data/', views.asset_category_performance_data, name='asset_category_performance_data'),
    path('transactions/', views.transactions, name='transactions'),
    path('add_transaction/', views.add_transaction, name='add_transaction'),
    path('fetch-chart-data/<str:ticker>/', views.fetch_chart_data, name='fetch_chart_data'),
    path('fetch-details/<str:ticker>/', views.fetch_details, name='fetch_details'),
    path('get_chartBig1_data/', views.get_chartBig1_data, name='get_chartBig1_data'),
    path('get_chartpurple_data/', views.get_chartpurple_data, name='get_chartpurple_data'),
    path('get_CountryChart_data/', views.get_CountryChart_data, name='get_CountryChart_data'),
    path('create_charts/', views.create_charts, name='create_charts'),
    path('create_charts_p/<str:ticker>/', views.create_charts_p, name='create_charts_p'),
    path('plot_prediction/', views.plot_prediction, name='plot_prediction'),
    path('get_ticker/', views.get_ticker, name='get_ticker'),
    # Matches any html file
    re_path(r'^.*\.*', views.pages, name='pages'),

]
