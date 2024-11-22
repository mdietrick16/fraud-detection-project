# Define the features to be selected for preprocessing
SELECTED_FEATURES = [
    'incident_severity',  # Severity of the incident
    'vehicle_claim',  # Amount claimed for vehicle damage
    'total_claim_amount',  # Total amount claimed
    'collision_type',  # Type of collision
    'property_claim',  # Amount claimed for property damage
    'injury_claim',  # Amount claimed for injuries
    'policy_csl',  # Combined single limit policy coverage
    'months_as_customer',  # Number of months as a customer
    'witnesses',  # Number of witnesses
    'age',  # Age of the policyholder
    'incident_hour_of_the_day',  # Hour of the day when the incident occurred
    'incident_state',  # State where the incident occurred
    'incident_city',  # City where the incident occurred
    'incident_type',  # Type of incident
    'insured_occupation'  # Occupation of the insured individual
]

# Define the target column for prediction
TARGET_COLUMN = 'fraud_reported'  # Target variable indicating if the claim is fraudulent
