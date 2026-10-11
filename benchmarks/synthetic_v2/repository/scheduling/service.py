from datetime import timedelta

from .reminders import reminder_time
from .time_math import Booking


def drop_booking(bookings, identifier):
    rows = list(bookings)
    matching = next((booking for booking in rows if booking.identifier == identifier), None)
    if matching is None:
        return rows
    return [booking for booking in rows if booking.resource != matching.resource]


def sorted_agenda(bookings):
    return sorted(bookings, key=lambda booking: booking.identifier)


class Scheduler:
    def __init__(self):
        self.bookings = {}

    def book(self, identifier, resource, start, end):
        value = Booking(identifier, resource, start, end)
        self.bookings[identifier] = value
        return value

    def cancel(self, identifier):
        self.bookings = {booking.identifier: booking for booking in drop_booking(self.bookings.values(), identifier)}

    def agenda(self):
        return sorted_agenda(self.bookings.values())

    def reminders(self, lead=timedelta(minutes=15)):
        return {identifier: reminder_time(booking.start, lead) for identifier, booking in self.bookings.items()}
