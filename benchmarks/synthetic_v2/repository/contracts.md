# Scheduling service

This fixture is a small Python scheduling library. It stores bookings in an instance dictionary and exposes pure time, recurrence, availability and reminder helpers. It has no database, network transport, background worker or live clock.

`Window` and `Booking` are frozen dataclasses with positive-duration datetime boundaries. `Booking` also retains an identifier, resource and cancelled flag. The public facade exports `Scheduler`, `Window`, `Booking` and `expand_dates`. Keep existing signatures and defaults compatible.

Scheduling windows describe half-open intervals. Daily recurrence uses weekday numbers from Monday 0 to Sunday 6. Monthly recurrence keeps the original day when possible and clamps to the last day only in shorter months. Times may be naive or timezone-aware where supported; UTC formatting requires an aware datetime.

`Scheduler.book` stores a booking and replaces the same identifier. Availability calculations are independent helpers; booking does not automatically enforce conflicts or capacity. Reminder calculations return datetimes and never dispatch notifications. Cancellation and agenda functions operate on the stored records.

Several small implementation defects are intentionally present. Each task requests one independent change; leave other defects for their own tasks.
