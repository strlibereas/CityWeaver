🏙️ CityWeaver
Fantasy City Map Generator
________________________________________
🌆 Overview
CityWeaver is an interactive desktop app for creating game-ready fantasy city maps.
It focuses on organic, realistic road networks and hand-crafted layout logic rather than procedural randomness — perfect for world-builders, tabletop RPG designers, and game developers who want believable medieval-style cities.
CityWeaver builds natural road layouts, river crossings, and walled settlements that feel like they’ve evolved over centuries — not drawn by a computer grid.
Every map feels distinct, realistic, and ready for export to your favorite game engine or world-building workflow.
________________________________________
✨ Features
•	🛣️ Organic Road Network — Generates natural, winding roads inspired by @pboechat’s excellent roadgen foundation.
•	🌊 Rivers, Bridges & Coasts — Seamlessly integrates water bodies and automatically places bridges.
•	🧱 Walls & Gates — Add fortified city perimeters for a medieval aesthetic.
•	🏘️ Buildings & Facilities — Fills the map with buildings and civic zones that can be toggled interactively.
•	🎛️ Intuitive UI Controls — Easily tweak city features, density, or styles through the sidebar panels.
•	🖼️ Vector-Style Rendering — Crisp visuals at any zoom level with smooth panning and zooming.
•	📤 Export to PNG & PDF — Save your final city layout with preserved borders and clickable facilities in the PDF.
________________________________________
🧩 Installation
CityWeaver is written in Python and uses Pygame for rendering.
It has been tested on Windows 10/11 with Python 3.9+.
1. Clone or Download
git clone https://github.com/strlibereas/CityWeaver.git
cd CityWeaver
2. Create a Virtual Environment
python -m venv venv
3. Activate the Environment
venv\Scripts\activate
4. Install Dependencies
pip install -r requirements.txt
________________________________________
🚀 Running CityWeaver
After setup, launch the app with:
python city_weaver.py
This opens the main CityWeaver window and generates your first map.
________________________________________
🕹️ Controls
Action	Description
Left Click + Drag	Pan the map
Mouse Wheel	Zoom in/out
Sidebar (Left)	Toggle features: water, walls, facilities, etc.
Sidebar (Right)	Adjust parameters with sliders
Export Button	Save your map as exported_city.png and exported_city.pdf
Esc	Exit the application
💡 Each launch creates a new, unique city. Rerun the command for a fresh layout.
________________________________________
💾 Exporting Maps
Click Export to save:
•	PNG: High-resolution image of your current view
•	PDF: Same map with clickable facility names (great for documentation or printouts)
Both files are saved in your current working directory.
________________________________________
🧱 Building a Standalone Executable (Optional)
Create a Windows .exe for distribution:
pyinstaller city_weaver.py --onefile --windowed --name CityWeaver
You’ll find the executable in the dist folder after the build completes.
________________________________________
⚙️ Compatibility
•	✅ Windows 10/11 — fully tested
•	⚠️ macOS/Linux — should work (Pygame is cross-platform) but untested
________________________________________
🧭 Known Limitations
•	Currently tested only on Windows
•	Non-procedural: CityWeaver uses structured, rule-based placement for realistic layouts
•	Occasional redraw lag may occur on very large cities (depends on hardware and zoom level)
________________________________________
🪄 License
MIT License — free to use, modify, and distribute.
________________________________________
💬 Credits
This project builds upon and extends the outstanding open-source work of
@pboechat — roadgen
whose city road network generation provided the backbone for this visualization system.
Developed and adapted for interactive fantasy map creation with additional UI, export, and world-building features.
________________________________________
🏷️ GitHub Tags / Topics
fantasy-map-generator  
city-generator  
road-network  
roads  
pygame  
map-tools  
medieval-city  
worldbuilding  
game-dev-tools  
urban-planning  
________________________________________
💫 Project Tagline (for GitHub)
🏙️ Interactive fantasy city map generator featuring organic road networks, water systems, and PDF export tools for game developers.

