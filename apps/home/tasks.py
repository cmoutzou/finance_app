from celery import shared_task
from django.utils.timezone import now
from .models import Portfolio, PortfolioPerformance
from django.contrib.auth.models import User
from django.db import transaction
import logging
from datetime import datetime, date

@shared_task
def my_task(arg1, arg2):
    # Task logic here
    result = arg1 + arg2
    return result

logger = logging.getLogger(__name__)


@shared_task
def update_portfolio_performance():
    try:        
        users = User.objects.all().prefetch_related('portfolio_set')
        today_date = now().date()

        for user in users:
            portfolios = user.portfolio_set.all()
            total_value = sum(
                portfolio.total_quantity * portfolio.current_market_price for portfolio in portfolios
            )

            performance, created = PortfolioPerformance.objects.update_or_create(
                user=user,
                date=today_date,
                defaults={
                    'total_value': total_value,
                    'updated_at': now()
                }
            )
            if created:
                logger.info(f"Created new performance record for user {user.id} on {today_date}.")
            else:
                logger.info(f"Updated performance record for user {user.id} on {today_date}.")

    except Exception as e:
        logger.error(f"Error updating portfolio performance: {str(e)}")
        raise


@shared_task
def hourly_portfolio_snapshot():
    """
    Create or update hourly snapshots for each user's portfolio.
    """
    all_users = Portfolio.objects.values_list('user', flat=True).distinct()
    print("hourly_portfolio_snapshot executed at", now())
    for user_id in all_users:
        # Filter portfolios for the current user
        user_portfolios = Portfolio.objects.filter(user_id=user_id)
        
        # Aggregate total value for the user
        total_value = sum(portfolio.current_value for portfolio in user_portfolios)

        # Check if an entry for today already exists for this user
        today_entry = PortfolioPerformance.objects.filter(date=date.today(), user_id=user_id).first()

        if today_entry:
            # Update the existing record
            today_entry.total_value = total_value
            today_entry.updated_at = datetime.now()
            today_entry.save()
        else:
            # Create a new record
            PortfolioPerformance.objects.create(
                user_id=user_id,
                date=date.today(),
                total_value=total_value
            )
    return f"Hourly snapshot completed for {len(all_users)} users."