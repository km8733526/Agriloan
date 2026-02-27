// Map Integration using Leaflet.js

let map;
let marker;
let selectedLocation = null;

document.addEventListener('DOMContentLoaded', function() {
    // Initialize map only if on the application page
    if (document.getElementById('map')) {
        initializeMap();
    }
});

function initializeMap() {
    // Default center (Maharashtra, India)
    const defaultLat = 19.9975;
    const defaultLng = 73.7898;
    
    // Initialize map
    map = L.map('map').setView([defaultLat, defaultLng], 8);
    
    // Add tile layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 19
    }).addTo(map);
    
    // Add click event to map
    map.on('click', function(e) {
        placeMarker(e.latlng);
    });
    
    // Try to get user's current location
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            function(position) {
                const userLat = position.coords.latitude;
                const userLng = position.coords.longitude;
                map.setView([userLat, userLng], 12);
                
                // Add a blue circle to show user's current location
                L.circle([userLat, userLng], {
                    color: 'blue',
                    fillColor: '#30f',
                    fillOpacity: 0.2,
                    radius: 500
                }).addTo(map);
            },
            function(error) {
                console.log('Geolocation error:', error);
            }
        );
    }
    
    // Add search control (basic implementation)
    addSearchControl();
}

function placeMarker(latlng) {
    // Remove existing marker if any
    if (marker) {
        map.removeLayer(marker);
    }
    
    // Add new marker
    marker = L.marker(latlng).addTo(map);
    
    // Update coordinates in form
    document.getElementById('latitude').value = latlng.lat.toFixed(6);
    document.getElementById('longitude').value = latlng.lng.toFixed(6);
    
    // Store selected location
    selectedLocation = latlng;
    
    // Add popup to marker
    marker.bindPopup(`
        <div class="text-center">
            <p class="font-semibold">Land Location Selected</p>
            <p class="text-sm text-gray-600">Lat: ${latlng.lat.toFixed(6)}</p>
            <p class="text-sm text-gray-600">Lng: ${latlng.lng.toFixed(6)}</p>
        </div>
    `).openPopup();
    
    // Show success notification
    showMapNotification('Location marked successfully');
}

function addSearchControl() {
    // Create custom control
    const searchControl = L.Control.extend({
        options: {
            position: 'topright'
        },
        
        onAdd: function(map) {
            const container = L.DomUtil.create('div', 'leaflet-bar leaflet-control');
            container.style.backgroundColor = 'white';
            container.style.padding = '5px';
            container.style.borderRadius = '4px';
            
            container.innerHTML = `
                <div style="display: flex; align-items: center;">
                    <input type="text" id="mapSearch" placeholder="Search location..." 
                           style="padding: 5px; border: 1px solid #ccc; border-radius: 4px; width: 200px;">
                    <button onclick="searchLocation()" 
                            style="margin-left: 5px; padding: 5px 10px; background: #16a34a; color: white; border: none; border-radius: 4px; cursor: pointer;">
                        Search
                    </button>
                </div>
            `;
            
            // Prevent click propagation to map
            L.DomEvent.disableClickPropagation(container);
            
            return container;
        }
    });
    
    map.addControl(new searchControl());
}

function searchLocation() {
    const searchTerm = document.getElementById('mapSearch').value;
    
    if (!searchTerm) {
        showMapNotification('Please enter a location to search', 'error');
        return;
    }
    
    // Use Nominatim API for geocoding
    fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(searchTerm)}&limit=1`)
        .then(response => response.json())
        .then(data => {
            if (data && data.length > 0) {
                const result = data[0];
                const lat = parseFloat(result.lat);
                const lng = parseFloat(result.lon);
                
                map.setView([lat, lng], 14);
                placeMarker(L.latLng(lat, lng));
                
                showMapNotification(`Location found: ${result.display_name}`);
            } else {
                showMapNotification('Location not found', 'error');
            }
        })
        .catch(error => {
            console.error('Search error:', error);
            showMapNotification('Error searching location', 'error');
        });
}

function showMapNotification(message, type = 'success') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `fixed top-20 right-4 ${type === 'error' ? 'bg-red-500' : 'bg-green-500'} text-white px-4 py-2 rounded shadow-lg z-50`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// Add ability to draw polygon for land boundary (advanced feature)
function initializeDrawingTools() {
    // Add drawing controls
    const drawnItems = new L.FeatureGroup();
    map.addLayer(drawnItems);
    
    const drawControl = new L.Control.Draw({
        edit: {
            featureGroup: drawnItems
        },
        draw: {
            polygon: {
                shapeOptions: {
                    color: '#16a34a',
                    fillColor: '#22c55e',
                    fillOpacity: 0.3
                }
            },
            polyline: false,
            rectangle: false,
            circle: false,
            marker: false,
            circlemarker: false
        }
    });
    
    map.addControl(drawControl);
    
    // Handle polygon creation
    map.on(L.Draw.Event.CREATED, function(event) {
        const layer = event.layer;
        drawnItems.addLayer(layer);
        
        // Calculate area
        if (layer instanceof L.Polygon) {
            const area = L.GeometryUtil.geodesicArea(layer.getLatLngs()[0]);
            const acres = (area * 0.000247105).toFixed(2);
            
            layer.bindPopup(`Land Area: ${acres} acres`).openPopup();
            
            // Update land size field
            const landSizeField = document.getElementById('landSize');
            if (landSizeField) {
                landSizeField.value = acres;
            }
        }
    });
}
