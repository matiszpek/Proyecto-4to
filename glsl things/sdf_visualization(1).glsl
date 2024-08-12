#ifdef GL_ES
precision mediump float;
#endif

uniform vec2 u_resolution;
uniform float u_time;

float sdf_sphere(vec3 p, float r) {
    return length(p) - r;
}

float sdf_box(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0);
}

float op_union(float d1, float d2) {
    return min(d1, d2);
}

float map(vec3 p) {
    float sphere = sdf_sphere(p - vec3(0.0, 0.0, 0.0), 0.3); // Centro en el origen
    float box = sdf_box(p - vec3(0.25, 0.0, 0.0), vec3(0.3)); // Cerca del origen
    return op_union(sphere, box);
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
    vec3 ro = vec3(0, 0, 2); // Posición de la cámara
    vec3 rd = normalize(vec3(uv, -1)); // Dirección de la cámara

    float t = 0.0;
    for (int i = 0; i < 256; i++) { // Aumentamos a 256 iteraciones
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
        if(t > 10.0) break; // Salimos del bucle si t es demasiado grande
    }

    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); // Color de fondo
}
