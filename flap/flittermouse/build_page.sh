#!/bin/sh
# index.html = _head.html + _body.html
cd "$(dirname "$0")" && cat _head.html _body.html > index.html
