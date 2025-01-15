from django import forms
from .models import Transaction

class TransactionForm(forms.ModelForm):
    class Meta:
        model = Transaction
        fields = ['ticker', 'transaction_type','name', 'asset_type', 'buy_price', 'number_assets']  # Correct field names
        widgets = {
            'ticker': forms.TextInput(attrs={'class': 'form-control'}),
            'transaction_type': forms.Select(attrs={'class': 'form-control'}),
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'asset_type': forms.TextInput(attrs={'class': 'form-control'}),#forms.Select(attrs={'class': 'form-control'}),
            'buy_price': forms.NumberInput(attrs={'class': 'form-control'}),
            'number_assets': forms.NumberInput(attrs={'class': 'form-control'}),
        }