from django.db import models
from django.contrib.auth.models import User
from django.utils.timezone import now


class Transaction(models.Model):
    ASSET_TYPES = [
        ('Stock', 'Stock'),
        ('Bond', 'Bond'),
        ('Derivative', 'Derivative'),
        ('ETF', 'ETF'),
        ('Mutual Fund', 'Mutual Fund'),
    ]
    TRANSACTION_TYPES = [
        ('Buy', 'Buy'),
        ('Sell', 'Sell'),
    ]
    transaction_type = models.CharField(
        max_length=100,
        choices=TRANSACTION_TYPES,
        default='Buy'
    )
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE,
        related_name='transactions'
    )
    ticker = models.CharField(max_length=10)
    name = models.CharField(max_length=100, default='N/A')
    asset_type = models.CharField(max_length=100, default='N/A') #models.CharField(max_length=100, choices=ASSET_TYPES)
    buy_price = models.DecimalField(max_digits=10, decimal_places=4, default=0)
    number_assets = models.DecimalField(max_digits=10, decimal_places=0, default=0)
    date = models.DateTimeField(default=now)

    class Meta:
        db_table = 'transactions'
        ordering = ['-date']

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        self.update_portfolio()


    def update_portfolio(self):
        portfolio, created = Portfolio.objects.get_or_create(
            user=self.user,
            ticker=self.ticker,
            defaults={'name': self.name, 'asset_type': self.asset_type}
        )

        if self.transaction_type == 'Buy':
            portfolio.total_quantity += self.number_assets
            portfolio.total_investment += self.number_assets * self.buy_price
        elif self.transaction_type == 'Sell':
            portfolio.total_quantity -= self.number_assets
            portfolio.total_investment -= self.number_assets * portfolio.average_buy_price

        portfolio.average_buy_price = (portfolio.total_investment / portfolio.total_quantity) if portfolio.total_quantity > 0 else 0
        portfolio.current_value = portfolio.total_quantity * portfolio.current_market_price
        portfolio.profit_loss = portfolio.current_value - portfolio.total_investment
        portfolio.save()

    def __str__(self):
        return f"{self.name} ({self.ticker})"
    

class Portfolio(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    ticker = models.CharField(max_length=10, default='N/A')
    name = models.CharField(max_length=100, default='N/A')
    asset_type = models.CharField(max_length=100, default="Stock")
    total_quantity = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    average_buy_price = models.DecimalField(max_digits=10, decimal_places=4, default=0)
    total_investment = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    current_market_price = models.DecimalField(max_digits=10, decimal_places=4, default=0)
    current_value = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    profit_loss = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'portfolio'
        unique_together = ('user', 'ticker')  # To ensure uniqueness per user per ticker

    def __str__(self):
        return f"{self.name} ({self.ticker})"


class PortfolioPerformance(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, default=1)
    date = models.DateField()
    total_value = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'portfolio_history'

    def __str__(self):
        return str(self.date)
    

class AssetCategoryPerformance(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    asset_type = models.CharField(max_length=100)
    date = models.DateField()
    total_value = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'asset_category_performance'
        unique_together = ('user', 'asset_type', 'date')  # Avoid duplicate entries

    def __str__(self):
        return f"{self.asset_type} - {self.date}"