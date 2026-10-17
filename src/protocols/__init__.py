# src/protocols/__init__.py

# این ایمپورت‌ها باعث می‌شوند وقتی پکیج لود می‌شود، 
# دکوریتور @register_protocol اجرا شده و کلاس‌ها ثبت شوند.

from .base import NetworkProtocol
from .factory import ProtocolFactory
from .names import ProtocolNames

# ایمپورت ماژول‌های حاوی پروتکل (برای ثبت خودکار)
from . import generic
from . import ironforge
# from . import dbfl  (اگر وجود دارد)