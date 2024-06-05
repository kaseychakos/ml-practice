from flask import jsonify

def validate(req):
    """Validate user input.

    Args:
        req (request): Flask request object

    Returns:
        Boolean
    """
    # check Content-Type and only accept application/json
    content_type = req.headers.get('Content-Type')
    if content_type != 'application/json':
        return False, jsonify({ 400: "incorrect Content-Type, accepts application/json"})
    
    # check json is provided
    data = req.get_json()
    if not data:
        return False, jsonify({ 400: "no JSON provided"})
    
    # check keys contain expected values
    expected_keys = ['query']
    
    for k in expected_keys:
        if k not in data:
            return False, jsonify({ 400: f'missing expected key: {k}'})
    
    for k,v in data.items():
        if not isinstance(v, list):
            return False, jsonify({ 400: f'expected a list of strings for key: {k}'})

        for i in v:
            if not isinstance(i, str):
                return jsonify({ 400: f'value is not a string for key: {k}'})

    return True, jsonify({ 200: 'success'})    