"""
Render the custom spacecraft scenario used for the thesis experiments.

The script follows the standard CORTO tutorial layout: it loads the scene,
geometry, and body assets from `input/<scenario_name>/`, initializes the camera,
Sun, body, rendering engine, and compositor branches, then renders the requested
number of image/label pairs.
"""

import sys
import os

sys.path.append(os.getcwd())
import cortopy as corto

# Clean all default objects before CORTO builds the scene.
corto.Utils.clean_scene()

# ============================================================
# Input files
# ============================================================

scenario_name = "S10_Spacecraft_Complex_Sat"
scene_name = "scene_CUSTOM.json"
geometry_name = "geometry.json"
body_name = "Jason2_rescaled.obj"

State = corto.State(scene = scene_name, geometry = geometry_name, body = body_name, scenario = scenario_name)

# Optional material input.
#State.add_path("material_name", os.path.join('input',scenario_name, 'body','material','shading_D1_S05_Didymos.json'))

# ============================================================
# Scene setup
# ============================================================

cam = corto.Camera('WFOV_Camera', State.properties_cam)
sun = corto.Sun('Sun',State.properties_sun)
name, extension = os.path.splitext(body_name)
body = corto.Body(name,State.properties_body)
rendering_engine = corto.Rendering(State.properties_rendering)
ENV = corto.Environment(cam, body, sun, rendering_engine)

# ============================================================
# Material and compositor setup
# ============================================================

#material = corto.Shading.load_material('S05_Didymos_Milani_material', State)
#corto.Shading.assign_material_to_object(material, body)

tree = corto.Compositing.create_compositing()
render_node = corto.Compositing.rendering_node(tree, (0,0))
#corto.Compositing.create_img_denoise_branch(tree,render_node)
corto.Compositing.create_depth_branch(tree,render_node)
corto.Compositing.create_slopes_branch(tree,render_node,State)

# ============================================================
# Rendering
# ============================================================

n_img = 100
for idx in range(0,n_img):
    ENV.PositionAll(State,index=idx)
    ENV.RenderOne(cam, State, index=idx, depth_flag = True)

# Save the configured .blend scene for inspection/debugging.
corto.Utils.save_blend(State)
