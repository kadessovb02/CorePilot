# Inventory fixture

This small Python package is a synthetic, in-memory inventory service. It has no HTTP transport,
database, persistence layer, or external dependencies. Client functions prepare transport data;
they do not send requests or sleep. Event records remain in the service instance.

The public facade exports `Product`, `CatalogService`, and `RemoteError`. Helper APIs are available
from their individual modules. Product prices use integer cents; loader price inputs use major
currency units. A Product is a frozen dataclass. Reservation returns a new Product.

Search returns a list in input order. Pagination uses a page number and page size. Event payloads
have an `id`, one of the kinds `created` or `restocked`, and optional additional JSON data. The
service checks an event before recording its JSON representation or marking its identifier seen.

Tasks describe the specific behavior to change. Keep other public interfaces and existing correct
behavior intact. This fixture intentionally contains independent small defects.
