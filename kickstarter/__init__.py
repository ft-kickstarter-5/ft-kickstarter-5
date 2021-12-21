import os
from .app import create_app

if __name__ == "__main__":
    APP = create_app()
    port = int(os.environ.get("PORT", 5000))
    APP.run(host="0.0.0.0", port=port)
else:
    APP = create_app()
