#ifdef GL_ES
precision mediump float;
#endif

uniform vec2 u_resolution;
uniform float u_time;

float sdf_sphere(vec3 p, float r) {
    return length(p) - r;
}

float map(vec3 p) {
    return sdf_sphere(p, 0.5);
}

vec3 getNormal(vec3 p) {
    float h = 0.0001;
    vec3 n;
    n.x = map(p + vec3(h, 0, 0)) - map(p - vec3(h, 0, 0));
    n.y = map(p + vec3(0, h, 0)) - map(p - vec3(0, h, 0));
    n.z = map(p + vec3(0, 0, h)) - map(p - vec3(0, 0, h));
    return normalize(n);
}

void main() {
    vec2 uv = (gl_FragCoord.xy - u_resolution.xy * 0.5) / u_resolution.y;
    vec3 ro = vec3(0, 0, 2); // Camera position
    vec3 rd = normalize(vec3(uv, -1)); // Camera direction

    float t = 0.0;
    for (int i = 0; i < 100; i++) {
        vec3 p = ro + t * rd;
        float d = map(p);
        if (d < 0.001) {
            vec3 n = getNormal(p);
            vec3 light_dir = normalize(vec3(1, 1, 1));
            float diff = max(dot(n, light_dir), 0.0);
            gl_FragColor = vec4(diff, diff, diff, 1.0);
            return;
        }
        t += d;
    }

    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
}
