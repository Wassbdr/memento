import { useEffect, useMemo, useRef, Suspense, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { EffectComposer, Bloom } from "@react-three/postprocessing";
import * as THREE from "three";

export type SphereState = "idle" | "listening" | "speaking";

const TAU = Math.PI * 2;
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));
const SPHERE_RADIUS = 1.54;

const POINT_VERT = /* glsl */ `
  const float TAU = 6.28318530718;

  uniform float uTime;
  uniform float uDeform;
  uniform float uPulse;
  uniform float uGlow;
  uniform vec3 uBands;
  uniform float uLevel;

  attribute float aBaseSize;
  attribute float aSeed;
  attribute vec3 aNormal;

  varying vec3 vColor;
  varying float vAlpha;

  void main() {
    vec3 pos = position;

    float lowMask = 1.0 - smoothstep(-0.65, -0.05, aNormal.y);
    float midMask = 1.0 - smoothstep(0.08, 0.62, abs(aNormal.y));
    vec2 highDir = normalize(vec2(cos(0.85), sin(0.85)));
    vec2 azDir = normalize(aNormal.xy + vec2(0.0001, 0.0));
    float highAz = max(dot(azDir, highDir), 0.0);
    float highMask = smoothstep(0.35, 0.92, highAz) * smoothstep(-0.15, 0.75, aNormal.y);

    float bandActivation = uBands.x * lowMask + uBands.y * midMask + uBands.z * highMask;
    float reactive = clamp(bandActivation + uLevel * 0.7, 0.0, 1.7);

    float waveA = sin(dot(aNormal, vec3(1.35, -0.85, 1.1)) * 5.2 - uTime * (0.65 + uPulse * 0.9));
    float waveB = sin(dot(aNormal, vec3(-0.7, 1.45, 0.55)) * 7.4 + uTime * (0.5 + uPulse * 0.35));
    float carrier = 0.5 + 0.5 * sin(uTime * (1.3 + reactive * 2.5) + aSeed * 1.2);
    float surface = (waveA * 0.58 + waveB * 0.42) * (uDeform * (0.004 + carrier * (0.012 + reactive * 0.018)));

    float az = atan(aNormal.y, aNormal.x);
    float ridges = sin(az * (14.0 + uBands.y * 10.0) + uTime * (2.0 + uBands.z * 6.0));
    surface += ridges * reactive * 0.006;

    pos += aNormal * surface;

    vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
    gl_Position = projectionMatrix * mvPosition;

    float sizeBoost = 1.0 + uGlow * 0.18 + carrier * reactive * 0.32 + uLevel * 0.18;
    gl_PointSize = (aBaseSize * sizeBoost * 520.0) / max(-mvPosition.z, 0.001);

    vec3 deepBlue = vec3(0.18, 0.32, 0.98);
    vec3 violet = vec3(0.55, 0.34, 1.00);
    vec3 magenta = vec3(0.95, 0.40, 0.95);

    float azMix = 0.5 + 0.5 * sin(az + 0.6);
    vec3 base = mix(deepBlue, violet, azMix);
    float rimTint = pow(1.0 - abs(aNormal.z), 1.6);
    base = mix(base, magenta, rimTint * 0.28 + uPulse * 0.14);

    vec3 lowColor = vec3(0.22, 0.42, 1.00) * (uBands.x * lowMask);
    vec3 midColor = vec3(0.62, 0.42, 1.00) * (uBands.y * midMask);
    vec3 highColor = vec3(1.00, 0.56, 0.90) * (uBands.z * highMask);
    vec3 spectralTint = lowColor + midColor + highColor;

    vec3 viewNormal = normalize(normalMatrix * aNormal);
    vec3 viewDir = normalize(-mvPosition.xyz);
    float fresnel = pow(1.0 - max(dot(viewNormal, viewDir), 0.0), 2.2);

    vColor = base * (0.88 + uGlow * 0.22) + spectralTint * 0.55 + fresnel * vec3(0.2, 0.28, 0.72);
    vAlpha = clamp(0.44 + fresnel * 0.62 + uGlow * 0.14 + reactive * 0.22, 0.0, 1.8);
  }
`;

const POINT_FRAG = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;

  void main() {
    vec2 uv = gl_PointCoord - 0.5;
    float dist = length(uv);

    float core = exp(-dist * dist * 20.0);
    float halo = exp(-dist * dist * 4.6) * 0.58;
    float edgeFade = 1.0 - smoothstep(0.5, 0.72, dist);
    float alpha = (core + halo) * vAlpha * edgeFade;
    if (alpha < 0.002) discard;
    vec3 col = vColor * (core * 1.2 + halo * 0.74);

    gl_FragColor = vec4(col, alpha);
  }
`;

interface SphereUniforms {
  uTime: THREE.IUniform<number>;
  uDeform: THREE.IUniform<number>;
  uPulse: THREE.IUniform<number>;
  uGlow: THREE.IUniform<number>;
  uBands: THREE.IUniform<THREE.Vector3>;
  uLevel: THREE.IUniform<number>;
}

interface SphereBuffers {
  positions: Float32Array;
  sizes: Float32Array;
  normals: Float32Array;
  seeds: Float32Array;
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function smoothAttackRelease(
  current: number,
  target: number,
  attack: number,
  release: number,
): number {
  const coeff = target > current ? attack : release;
  return current + (target - current) * coeff;
}

function getBandAverage(
  data: Uint8Array,
  sampleRate: number,
  fromHz: number,
  toHz: number,
): number {
  const nyquist = sampleRate * 0.5;
  const hzPerBin = nyquist / data.length;
  const start = Math.max(0, Math.floor(fromHz / hzPerBin));
  const end = Math.min(data.length - 1, Math.floor(toHz / hzPerBin));

  if (end <= start) return 0;

  let sum = 0;
  for (let i = start; i <= end; i += 1) {
    sum += data[i];
  }

  return sum / (end - start + 1) / 255;
}

function buildSphereBuffers(pointCount: number): SphereBuffers {
  const positions = new Float32Array(pointCount * 3);
  const sizes = new Float32Array(pointCount);
  const normals = new Float32Array(pointCount * 3);
  const seeds = new Float32Array(pointCount);

  for (let i = 0; i < pointCount; i += 1) {
    const t = (i + 0.5) / pointCount;
    const phi = Math.acos(1 - 2 * t);
    const theta = i * GOLDEN_ANGLE;

    const nx = Math.sin(phi) * Math.cos(theta);
    const ny = Math.cos(phi);
    const nz = Math.sin(phi) * Math.sin(theta);

    const radius = SPHERE_RADIUS + (Math.random() * 2 - 1) * 0.015;
    const idx3 = i * 3;

    positions[idx3] = nx * radius;
    positions[idx3 + 1] = ny * radius;
    positions[idx3 + 2] = nz * radius;

    normals[idx3] = nx;
    normals[idx3 + 1] = ny;
    normals[idx3 + 2] = nz;

    sizes[i] = 0.018 + Math.random() * 0.01;
    seeds[i] = Math.random();
  }

  return { positions, sizes, normals, seeds };
}

// ─── Animated Sphere ─────────────────────────────────────────────────────────

interface AnimatedSphereProps {
  state: SphereState;
  reducedMotion: boolean;
}

function AnimatedSphere({ state, reducedMotion }: AnimatedSphereProps) {
  const groupRef = useRef<THREE.Group | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const freqDataRef = useRef<Uint8Array<ArrayBuffer> | null>(null);
  const timeDataRef = useRef<Uint8Array<ArrayBuffer> | null>(null);
  const bandStateRef = useRef(new THREE.Vector3(0, 0, 0));
  const levelStateRef = useRef(0);

  const uniforms = useMemo<SphereUniforms>(
    () => ({
      uTime: { value: 0 },
      uDeform: { value: 0.15 },
      uPulse: { value: 0.08 },
      uGlow: { value: 0.18 },
      uBands: { value: new THREE.Vector3(0, 0, 0) },
      uLevel: { value: 0 },
    }),
    [],
  );

  useEffect(() => {
    const wantsMic = state !== "idle";
    let cancelled = false;

    const stopCapture = () => {
      sourceRef.current?.disconnect();
      sourceRef.current = null;
      analyserRef.current = null;
      freqDataRef.current = null;
      timeDataRef.current = null;

      if (streamRef.current) {
        for (const track of streamRef.current.getTracks()) {
          track.stop();
        }
        streamRef.current = null;
      }

      if (audioCtxRef.current) {
        const ctx = audioCtxRef.current;
        audioCtxRef.current = null;
        void ctx.close();
      }

      bandStateRef.current.set(0, 0, 0);
      levelStateRef.current = 0;
    };

    const startCapture = async () => {
      if (!wantsMic || analyserRef.current) return;
      if (!navigator.mediaDevices?.getUserMedia) return;

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        });

        if (cancelled) {
          for (const track of stream.getTracks()) {
            track.stop();
          }
          return;
        }

        const audioCtx = new window.AudioContext();
        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 2048;
        analyser.smoothingTimeConstant = 0.84;

        const source = audioCtx.createMediaStreamSource(stream);
        source.connect(analyser);
        await audioCtx.resume();

        streamRef.current = stream;
        audioCtxRef.current = audioCtx;
        analyserRef.current = analyser;
        sourceRef.current = source;
        freqDataRef.current = new Uint8Array(new ArrayBuffer(analyser.frequencyBinCount));
        timeDataRef.current = new Uint8Array(new ArrayBuffer(analyser.fftSize));
      } catch {
        // Fallback animation remains active when mic access is denied.
      }
    };

    if (wantsMic) {
      void startCapture();
    } else {
      stopCapture();
    }

    return () => {
      cancelled = true;
      stopCapture();
    };
  }, [state]);

  const sphereCount = reducedMotion ? 5600 : 8600;

  const sphereGeo = useMemo(() => {
    const { positions, sizes, normals, seeds } = buildSphereBuffers(sphereCount);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("aBaseSize", new THREE.BufferAttribute(sizes, 1));
    geometry.setAttribute("aNormal", new THREE.BufferAttribute(normals, 3));
    geometry.setAttribute("aSeed", new THREE.BufferAttribute(seeds, 1));
    return geometry;
  }, [sphereCount]);

  const mat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        uniforms: uniforms as unknown as { [key: string]: THREE.IUniform },
        vertexShader: POINT_VERT,
        fragmentShader: POINT_FRAG,
        transparent: true,
        depthWrite: false,
        depthTest: true,
        toneMapped: false,
        blending: THREE.AdditiveBlending,
      }),
    [uniforms],
  );

  useEffect(() => () => sphereGeo.dispose(), [sphereGeo]);
  useEffect(() => () => mat.dispose(), [mat]);

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    uniforms.uTime.value = t;

    const isListening = state === "listening";
    const isSpeaking = state === "speaking";
    const isActive = isListening || isSpeaking;

    const analyser = analyserRef.current;
    const freqData = freqDataRef.current;
    const timeData = timeDataRef.current;

    let low = 0;
    let mid = 0;
    let high = 0;
    let level = 0;

    if (analyser && freqData && timeData && audioCtxRef.current) {
      analyser.getByteFrequencyData(freqData);
      analyser.getByteTimeDomainData(timeData);

      const sampleRate = audioCtxRef.current.sampleRate;
      const lowBand = getBandAverage(freqData, sampleRate, 60, 320);
      const midBand = getBandAverage(freqData, sampleRate, 320, 2200);
      const highBand = getBandAverage(freqData, sampleRate, 2200, 7000);

      low = clamp01((lowBand - 0.03) * 2.6);
      mid = clamp01((midBand - 0.02) * 2.4);
      high = clamp01((highBand - 0.012) * 3.1);

      low = Math.pow(low, 0.95);
      mid = Math.pow(mid, 0.98);
      high = Math.pow(high, 1.08);

      let sumSquares = 0;
      for (let i = 0; i < timeData.length; i += 1) {
        const sample = (timeData[i] - 128) / 128;
        sumSquares += sample * sample;
      }
      const rms = Math.sqrt(sumSquares / timeData.length);
      level = clamp01((rms - 0.015) * 6.8);
      level = Math.pow(level, 0.92);
    } else if (isListening || isSpeaking) {
      const fallback = isSpeaking
        ? 0.55 + 0.35 * (0.5 + 0.5 * Math.sin(t * 2.4))
        : 0.25 + 0.2 * (0.5 + 0.5 * Math.sin(t * 1.6));
      low = fallback * (0.75 + 0.25 * Math.sin(t * 1.1 + 0.5));
      mid = fallback * (0.65 + 0.35 * Math.sin(t * 1.6 + 1.3));
      high = fallback * (0.55 + 0.45 * Math.sin(t * 2.7 + 2.4));
      level = fallback * 0.7;
    }

    const bands = bandStateRef.current;
    bands.x = smoothAttackRelease(bands.x, low, 0.1, 0.04);
    bands.y = smoothAttackRelease(bands.y, mid, 0.1, 0.04);
    bands.z = smoothAttackRelease(bands.z, high, 0.11, 0.045);
    levelStateRef.current = smoothAttackRelease(levelStateRef.current, level, 0.12, 0.05);

    const spectral = bands.x * 0.4 + bands.y * 0.58 + bands.z * 0.7;
    const idleBreath = 0.5 + 0.5 * Math.sin(t * 0.42);
    const idleLevel = 0.035 + idleBreath * 0.02;

    const activity = isActive ? spectral + levelStateRef.current * 0.45 : idleLevel;

    const motionFactor = reducedMotion ? 0.45 : 1;
    const targetDeform = clamp01(0.05 + activity * 0.72 * motionFactor);
    const targetPulse = clamp01(0.04 + activity * 0.8 * motionFactor);
    const targetGlow = clamp01(0.13 + activity * 0.58 * motionFactor);

    uniforms.uBands.value.lerp(bands, 0.1);
    uniforms.uLevel.value += (levelStateRef.current - uniforms.uLevel.value) * 0.11;

    uniforms.uDeform.value += (targetDeform - uniforms.uDeform.value) * (reducedMotion ? 0.02 : 0.03);
    uniforms.uPulse.value += (targetPulse - uniforms.uPulse.value) * (reducedMotion ? 0.025 : 0.04);
    uniforms.uGlow.value += (targetGlow - uniforms.uGlow.value) * (reducedMotion ? 0.022 : 0.035);

    if (groupRef.current) {
      const sway = isActive || reducedMotion ? 0 : Math.sin(t * 0.36) * 0.01;
      groupRef.current.rotation.x = sway * 0.55;
      groupRef.current.rotation.z = sway * 0.35;

      const breathSpeed = isActive ? 0.55 + uniforms.uPulse.value * 1.1 : 0.42;
      const breathAmp = isActive
        ? 0.0018 + uniforms.uPulse.value * (reducedMotion ? 0.0022 : 0.0046)
        : 0.0032 + uniforms.uPulse.value * (reducedMotion ? 0.0008 : 0.0016);
      const breath =
        1 + Math.sin(t * breathSpeed) * breathAmp;
      groupRef.current.scale.setScalar(breath);
    }
  });

  return (
    <group ref={groupRef}>
      <points geometry={sphereGeo} material={mat} frustumCulled={false} />
    </group>
  );
}

// ─── Public Component ─────────────────────────────────────────────────────────

interface OrganicSphereProps {
  state?: SphereState;
}

export default function OrganicSphere({ state = "idle" }: OrganicSphereProps) {
  const [reducedMotion, setReducedMotion] = useState(false);
  const [compactViewport, setCompactViewport] = useState(false);

  useEffect(() => {
    const mqMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    const mqCompact = window.matchMedia("(max-width: 900px)");

    const sync = () => {
      setReducedMotion(mqMotion.matches);
      setCompactViewport(mqCompact.matches);
    };

    sync();

    const add = (mql: MediaQueryList, handler: () => void) => {
      if (mql.addEventListener) {
        mql.addEventListener("change", handler);
      } else {
        mql.addListener(handler);
      }
    };

    const remove = (mql: MediaQueryList, handler: () => void) => {
      if (mql.removeEventListener) {
        mql.removeEventListener("change", handler);
      } else {
        mql.removeListener(handler);
      }
    };

    add(mqMotion, sync);
    add(mqCompact, sync);

    return () => {
      remove(mqMotion, sync);
      remove(mqCompact, sync);
    };
  }, []);

  const dpr = useMemo<[number, number]>(() => {
    const pixelRatio = window.devicePixelRatio || 1;
    const cap = compactViewport ? 1.45 : 1.9;
    return [1, Math.min(pixelRatio, cap)];
  }, [compactViewport]);

  const bloomIntensity = reducedMotion ? 1.35 : compactViewport ? 2.0 : 2.75;

  return (
    <Canvas
      camera={{ position: [0, 0, 5.3], fov: 42 }}
      gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
      dpr={dpr}
      performance={{ min: 0.75 }}
      style={{ background: "transparent", width: "100%", height: "100%" }}
      onCreated={({ gl }) => {
        gl.setClearColor(0x000000, 0);
      }}
    >
      <Suspense fallback={null}>
        <AnimatedSphere state={state} reducedMotion={reducedMotion} />

        <EffectComposer>
          <Bloom
            luminanceThreshold={0.02}
            luminanceSmoothing={0.2}
            intensity={bloomIntensity}
            mipmapBlur
          />
        </EffectComposer>
      </Suspense>
    </Canvas>
  );
}
