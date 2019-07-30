<%
    
    #noise = opts.get("noise", False)
    #if noise:
    #    import numpy as np
    #    track_width += np.random.uniform(-1, 1)

    cart_friction = 0.0005
    pole_friction = 0.000002

    # TODO: remove, I'm  overwrititng the action space either way in StandEnvVime
    from solenoid.envs import constants
    ctrl_lim_low = float(constants.action_low)
    ctrl_lim_high = float(constants.action_high)

    height_low = constants.height_max
    height_high = constants.height_min
    track_width = height_high - height_low    

    # // Set the gravity scale to zero so this body will float
    #bodyDef.gravityScale = 0.0f;
%>


<box2d>
  <world timestep="0.02"> 
    <body name="weights" type="dynamic" position="0,0.00">
      <fixture density="5" friction="${cart_friction}" shape="polygon" box="0.1, 0.02"/>
    </body>
    <body name="track" type="static" position="0,${-track_width}">
      <fixture friction="${pole_friction}" group="-1" shape="polygon" box="0.02, ${track_width}"/>
    </body>
    <body name="ticks" type="static" position="0.0, 0.25">
      <fixture group="-1" shape="polygon" box="0.15,0.02"/>
    </body>
    <body name="ticks" type="static" position="0.0,0.5">
      <fixture group="-1" shape="polygon" box="0.15,0.02"/>
    </body>
    <body name="ticks" type="static" position="0.0,0.75">
      <fixture group="-1" shape="polygon" box="0.15,0.02"/>
    </body>

    <state type="xpos" body="weights"/> <!-- Height -->
    <state type="xvel" body="weights"/> <!-- Height Rate -->

    <control type="force" body="weights" anchor="0,0" direction="1,0" ctrllimit="${ctrl_lim_low},${ctrl_lim_high}"/>
  </world>
</box2d>
