def geo_to_pixel(locations, center):
    """
    Convert geographical coordinates to pixel coordinates for Sentinel-2 imagery.
    
    Assumes:
    - Image size: 512 x 512 pixels
    - Ground Sampling Distance: 10 meters per pixel (Sentinel-2 standard)
    - Image centered at provided center coordinates
    """
    import math
    
    height = 512
    width = 512
    gsd_meters = 10.0  # Sentinel-2 resolution
    
    center_lat, center_lon = center
    
    # Meters per degree at center latitude
    meters_per_degree_lat = 111320.0
    meters_per_degree_lon = 111320.0 * math.cos(math.radians(center_lat))
    
    pixel_locations = {}
    
    for loc_name, coords in locations.items():
        lat, lon = coords
        
        # Calculate offset from center in degrees, then meters
        lon_diff = lon - center_lon
        lat_diff = lat - center_lat
        
        x_meters = lon_diff * meters_per_degree_lon
        y_meters = lat_diff * meters_per_degree_lat
        
        # Convert to pixels (note: Y is flipped for image coordinates)
        x_pixel = int((x_meters / gsd_meters) + width / 2)
        y_pixel = int((-y_meters / gsd_meters) + height / 2)
        
        pixel_locations[loc_name] = (x_pixel, y_pixel)
    
    return pixel_locations