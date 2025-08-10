"""This module contains utility functions for the cognee."""

import os
import requests
from datetime import datetime, timezone
import networkx as nx
import matplotlib.pyplot as plt
import http.server
import socketserver
from threading import Thread
import pathlib
from uuid import uuid4

from cognee.base_config import get_base_config
from cognee.infrastructure.databases.graph import get_graph_engine


# Analytics Proxy Url, currently hosted by Vercel
proxy_url = "https://test.prometh.ai"


def get_entities(tagged_tokens):
    import nltk

    nltk.download("maxent_ne_chunker", quiet=True)

    from nltk.chunk import ne_chunk

    return ne_chunk(tagged_tokens)


def extract_pos_tags(sentence):
    """Extract Part-of-Speech (POS) tags for words in a sentence."""
    import nltk

    # Ensure that the necessary NLTK resources are downloaded
    nltk.download("words", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)

    from nltk.tag import pos_tag
    from nltk.tokenize import word_tokenize

    # Tokenize the sentence into words
    tokens = word_tokenize(sentence)

    # Tag each word with its corresponding POS tag
    pos_tags = pos_tag(tokens)

    return pos_tags


def get_anonymous_id():
    """Creates or reads a anonymous user id"""
    tracking_id = os.getenv("TRACKING_ID", None)

    if tracking_id:
        return tracking_id

    home_dir = str(pathlib.Path(pathlib.Path(__file__).parent.parent.parent.resolve()))

    if not os.path.isdir(home_dir):
        os.makedirs(home_dir, exist_ok=True)
    anonymous_id_file = os.path.join(home_dir, ".anon_id")
    if not os.path.isfile(anonymous_id_file):
        anonymous_id = str(uuid4())
        with open(anonymous_id_file, "w", encoding="utf-8") as f:
            f.write(anonymous_id)
    else:
        with open(anonymous_id_file, "r", encoding="utf-8") as f:
            anonymous_id = f.read()
    return anonymous_id


def send_telemetry(event_name: str, user_id, additional_properties: dict = {}):
    if os.getenv("TELEMETRY_DISABLED"):
        return

    env = os.getenv("ENV")
    if env in ["test", "dev"]:
        return

    current_time = datetime.now(timezone.utc)
    payload = {
        "anonymous_id": str(get_anonymous_id()),
        "event_name": event_name,
        "user_properties": {
            "user_id": str(user_id),
        },
        "properties": {
            "time": current_time.strftime("%m/%d/%Y"),
            "user_id": str(user_id),
            **additional_properties,
        },
    }

    response = requests.post(proxy_url, json=payload)

    if response.status_code != 200:
        print(f"Error sending telemetry through proxy: {response.status_code}")


def embed_logo(p, layout_scale, logo_alpha, position):
    """
    Embed a logo into the graph visualization as a watermark.
    """

    # svg_logo = """<svg width="1294" height="324" viewBox="0 0 1294 324" fill="none" xmlns="http://www.w3.org/2000/svg">
    #     <mask id="mask0_103_2579" style="mask-type:alpha" maskUnits="userSpaceOnUse" x="0" y="0" width="1294" height="324">
    #     <path fill-rule="evenodd" clip-rule="evenodd" d="M380.648 131.09C365.133 131.09 353.428 142.843 353.428 156.285V170.258C353.428 183.7 365.133 195.452 380.648 195.452C388.268 195.452 393.57 193.212 401.288 187.611C405.57 184.506 411.579 185.449 414.682 189.714C417.805 193.978 416.842 199.953 412.561 203.038C402.938 209.995 393.727 214.515 380.628 214.515C355.49 214.555 334.241 195.197 334.241 170.258V156.285C334.241 131.366 355.49 112.008 380.648 112.008C393.747 112.008 402.958 116.528 412.581 123.485C416.862 126.59 417.805 132.545 414.702 136.809C411.579 141.074 405.589 142.017 401.308 138.912C393.59 133.331 388.268 131.071 380.667 131.071L380.648 131.09ZM474.875 131.09C459.792 131.09 447.557 143.255 447.557 158.289V168.509C447.557 183.543 459.792 195.708 474.875 195.708C489.958 195.708 501.977 183.602 501.977 168.509V158.289C501.977 143.196 489.879 131.09 474.875 131.09ZM428.37 158.289C428.37 132.741 449.188 112.008 474.875 112.008C500.563 112.008 521.164 132.8 521.164 158.289V168.509C521.164 193.998 500.622 214.79 474.875 214.79C449.129 214.79 428.37 194.057 428.37 168.509V158.289ZM584.774 131.601C569.652 131.601 557.457 143.747 557.457 158.683C557.457 173.618 569.672 185.764 584.774 185.764C599.877 185.764 611.876 173.697 611.876 158.683C611.876 143.668 599.818 131.601 584.774 131.601ZM538.269 158.683C538.269 133.154 559.126 112.519 584.774 112.519C595.693 112.519 605.67 116.253 613.545 122.483L620.733 115.329C624.484 111.595 630.552 111.595 634.303 115.329C638.054 119.063 638.054 125.096 634.303 128.83L625.819 137.281C629.178 143.688 631.063 150.979 631.063 158.702C631.063 184.152 610.501 204.866 584.774 204.866C584.519 204.866 584.264 204.866 584.008 204.866H563.643C560.226 204.866 557.457 207.617 557.457 211.017C557.457 214.417 560.226 217.168 563.643 217.168H589.939H598.345C605.258 217.168 612.426 219.075 618.18 223.614C624.131 228.292 627.901 235.229 628.569 243.739C629.747 258.812 619.123 269.11 610.482 272.431L586.444 283.004C581.593 285.127 575.937 282.945 573.796 278.131C571.655 273.316 573.855 267.675 578.686 265.553L602.96 254.882C603.137 254.803 603.333 254.724 603.51 254.665C604.531 254.292 606.259 253.191 607.614 251.364C608.871 249.674 609.598 247.649 609.421 245.252C609.146 241.754 607.811 239.808 606.259 238.609C604.551 237.253 601.84 236.271 598.325 236.271H564.036C563.937 236.271 563.839 236.271 563.721 236.271H563.604C549.601 236.271 538.23 224.97 538.23 211.037C538.23 201.997 543.002 194.077 550.171 189.616C542.747 181.44 538.23 170.612 538.23 158.722L538.269 158.683ZM694.045 131.601C679.021 131.601 666.825 143.727 666.825 158.683V205.239C666.825 210.506 662.525 214.79 657.242 214.79C651.959 214.79 647.658 210.526 647.658 205.239V158.683C647.658 133.193 668.436 112.519 694.065 112.519C719.693 112.519 740.471 133.193 740.471 158.683V205.239C740.471 210.506 736.17 214.79 730.887 214.79C725.605 214.79 721.304 210.526 721.304 205.239V158.683C721.304 143.727 709.128 131.601 694.084 131.601H694.045ZM807.204 131.621C791.748 131.621 779.356 143.963 779.356 159.017V168.843C779.356 183.897 791.748 196.238 807.204 196.238C812.565 196.238 817.514 194.745 821.698 192.19C826.214 189.439 832.126 190.834 834.895 195.334C837.664 199.835 836.25 205.711 831.733 208.462C824.604 212.825 816.179 215.321 807.204 215.321C781.3 215.321 760.169 194.588 760.169 168.843V159.017C760.169 133.272 781.3 112.538 807.204 112.538C829.357 112.538 847.778 127.671 852.707 148.07L854.632 156.049L813.744 172.597C808.834 174.581 803.237 172.243 801.234 167.349C799.231 162.475 801.587 156.894 806.497 154.909L830.947 145.004C826.156 136.986 817.338 131.601 807.165 131.601L807.204 131.621ZM912.37 131.621C896.914 131.621 884.522 143.963 884.522 159.017V168.843C884.522 183.897 896.914 196.238 912.37 196.238C917.732 196.238 922.681 194.745 926.864 192.19C928.965 190.913 930.89 189.36 932.559 187.572C936.192 183.72 942.261 183.543 946.11 187.139C949.979 190.736 950.175 196.789 946.542 200.621C943.694 203.628 940.454 206.281 936.879 208.462C929.731 212.825 921.326 215.321 912.331 215.321C886.427 215.321 865.296 194.588 865.296 168.843V159.017C865.296 133.272 886.427 112.538 912.331 112.538C934.484 112.538 952.905 127.671 957.834 148.07L959.759 156.049L918.871 172.597C913.961 174.581 908.364 172.243 906.361 167.349C904.358 162.475 906.714 156.894 911.624 154.909L936.074 145.004C931.282 136.986 922.465 131.601 912.292 131.601L912.37 131.621Z" fill="#6510F4"/>
    #     </mask>
    #     <g mask="url(#mask0_103_2579)">
    #     <rect x="86" y="-119" width="1120" height="561" fill="#6510F4"/>
    #     </g>
    #     </svg>"""
    # Convert the SVG to a ReportLab Drawing

    logo_url = "data:image/png;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4KPCEtLSBHZW5lcmF0b3I6IEFkb2JlIElsbHVzdHJhdG9yIDI1LjQuMSwgU1ZHIEV4cG9ydCBQbHVnLUluIC4gU1ZHIFZlcnNpb246IDYuMDAgQnVpbGQgMCkgIC0tPgo8c3ZnIHZlcnNpb249IjEuMSIgaWQ9IkxheWVyXzEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHg9IjBweCIgeT0iMHB4IgoJIHZpZXdCb3g9IjAgMCAxMzUuNSAxMzUuNSIgc3R5bGU9ImVuYWJsZS1iYWNrZ3JvdW5kOm5ldyAwIDAgMTM1LjUgMTM1LjU7IiB4bWw6c3BhY2U9InByZXNlcnZlIj4KPHN0eWxlIHR5cGU9InRleHQvY3NzIj4KCS5zdDB7ZmlsbDp1cmwoI1NWR0lEXzFfKTt9Cjwvc3R5bGU+CjxnIGlkPSJfeDMwX2MxODUxMjMtNmM5ZC00ZGRjLTgyMjktYzdjZjRmODhiYTY1Ij4KCQoJCTxsaW5lYXJHcmFkaWVudCBpZD0iU1ZHSURfMV8iIGdyYWRpZW50VW5pdHM9InVzZXJTcGFjZU9uVXNlIiB4MT0iLTExNi4yMTkyIiB5MT0iMjQ2LjI5NjQiIHgyPSItMTE0LjM2NTUiIHkyPSIyNDYuMjk2NCIgZ3JhZGllbnRUcmFuc2Zvcm09Im1hdHJpeCg2NS4yNTQyIDI1LjM1NzEgMzAuNDI4NSAtNTQuMzc4NSA3My45NjM0IDE2Mzc4Ljc2MDcpIj4KCQk8c3RvcCAgb2Zmc2V0PSIwIiBzdHlsZT0ic3RvcC1jb2xvcjojOEU0MkQzIi8+CgkJPHN0b3AgIG9mZnNldD0iMSIgc3R5bGU9InN0b3AtY29sb3I6I0Q0Njk3NSIvPgoJPC9saW5lYXJHcmFkaWVudD4KCTxwYXRoIGNsYXNzPSJzdDAiIGQ9Ik0xMzMuNSw2Ni42YzAtMjQuNi0xOS45LTQ0LjUtNDQuNS00NC41Qzc5LjQsMTQuOSw2Ny42LDExLDU1LjYsMTFDMjQuOSwxMSwwLDM1LjksMCw2Ni42czI0LjksNTUuNiw1NS42LDU1LjYKCQljMTIsMCwyMy43LTMuOSwzMy40LTExLjFDMTEzLjYsMTExLjEsMTMzLjUsOTEuMiwxMzMuNSw2Ni42eiBNODcuMSwxMDUuNWMtNy41LTAuNC0xNC43LTIuOS0yMC44LTcuM0M3OS41LDkzLjgsODksODEuMyw4OSw2Ni42CgkJUzc5LjUsMzkuNSw2Ni4zLDM1YzYuMS00LjQsMTMuMy02LjksMjAuOC03LjNjMTEuMyw5LjIsMTguNiwyMy4yLDE4LjYsMzguOVM5OC40LDk2LjMsODcuMSwxMDUuNXogTTUwLjEsNjYuNgoJCWMwLTEwLjYsNC4zLTIwLjIsMTEuMi0yNy4yQzczLjksNDIsODMuNCw1My4yLDgzLjQsNjYuNnMtOS42LDI0LjYtMjIuMiwyNy4yQzU0LjMsODYuOSw1MC4xLDc3LjIsNTAuMSw2Ni42eiBNNTQuMiw5NC40CgkJYy0xNC43LTAuNy0yNi40LTEyLjktMjYuNC0yNy44czExLjctMjcsMjYuNC0yNy44Yy02LjMsNy45LTkuOCwxNy43LTkuNywyNy44QzQ0LjUsNzcuMiw0OC4xLDg2LjgsNTQuMiw5NC40eiBNNS42LDY2LjYKCQlDNS42LDM5LDI4LDE2LjYsNTUuNiwxNi42YzguOSwwLDE3LjMsMi4zLDI0LjUsNi40Yy03LjcsMS42LTE0LjksNS4yLTIwLjgsMTAuNWMtMS4yLTAuMS0yLjUtMC4yLTMuNy0wLjIKCQljLTE4LjQsMC0zMy40LDE0LjktMzMuNCwzMy40UzM3LjIsMTAwLDU1LjYsMTAwYzEuMywwLDIuNS0wLjEsMy43LTAuMmM1LjksNS4zLDEzLjEsOC45LDIwLjgsMTAuNWMtNy41LDQuMi0xNS45LDYuNS0yNC41LDYuNAoJCUMyOCwxMTYuNyw1LjYsOTQuMiw1LjYsNjYuNnogTTk1LjksMTA0LjljOS41LTEwLDE1LjMtMjMuNSwxNS4zLTM4LjNzLTUuOC0yOC4zLTE1LjMtMzguM2MxOC4yLDMuMywzMiwxOS4yLDMyLDM4LjMKCQlTMTE0LjEsMTAxLjcsOTUuOSwxMDQuOXoiLz4KPC9nPgo8L3N2Zz4K"

    # Finally, add the PNG image to your Bokeh plot as an image_url
    p.image_url(
        url=[logo_url],
        x=-layout_scale * 0.5,
        y=layout_scale * 0.5,
        w=layout_scale,
        h=layout_scale,
        anchor=position,
        global_alpha=logo_alpha,
    )


def start_visualization_server(
    host="0.0.0.0", port=8001, handler_class=http.server.SimpleHTTPRequestHandler
):
    """
    Spin up a simple HTTP server in a background thread to serve files.
    This is especially handy for quick demos or visualization purposes.

    Returns a shutdown() function that can be called to stop the server.

    :param host: Host/IP to bind to. Defaults to '0.0.0.0'.
    :param port: Port to listen on. Defaults to 8001.
    :param handler_class: A handler class, defaults to SimpleHTTPRequestHandler.
    :return: A no-argument function `shutdown` which, when called, stops the server.
    """
    # Create the server
    server = socketserver.TCPServer((host, port), handler_class)

    def _serve_forever():
        print(f"Visualization server running at: http://{host}:{port}")
        server.serve_forever()

    # Start the server in a background thread
    thread = Thread(target=_serve_forever, daemon=True)
    thread.start()

    def shutdown():
        """
        Shuts down the server and blocks until the thread is joined.
        """
        server.shutdown()  # Signals the serve_forever() loop to stop
        server.server_close()  # Frees up the socket
        thread.join()
        print(f"Visualization server on port {port} has been shut down.")

    # Return only the shutdown function (the server runs in the background)
    return shutdown
