# myapp/signals.py
from django.contrib.auth.signals import user_logged_in
from django.dispatch import receiver
from django_celery_beat.models import PeriodicTask, IntervalSchedule
from .tasks import update_database

@receiver(user_logged_in)
def start_task_on_login(sender, request, user, **kwargs):
    schedule, created = IntervalSchedule.objects.get_or_create(
        every=1,  # 1 minute
        period=IntervalSchedule.MINUTES,
    )

    # Create a periodic task for the user
    PeriodicTask.objects.get_or_create(
        interval=schedule,
        name=f'update-database-{user.id}',  # Unique name for each user
        task='myapp.tasks.update_database',
        args=[user.id],  # Pass user_id as an argument
    )