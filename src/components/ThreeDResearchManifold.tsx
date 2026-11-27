import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { Box, ChevronDown, ChevronUp, Compass, RotateCcw } from 'lucide-react';
import { FlagshipProject, SyntheticDataType } from '../types';
import { ML_TECHNIQUE_META, MlTechniqueId, SYNTHETIC_DATA_TYPE_META } from '../utils/syntheticDataset';

export type ThreeDGraphMode = 'loss_surface' | 'val_acc_pillars' | 'k9_clusters' | 'tensor_voxels';

interface EpochBarPoint {
  epoch: number;
  loss: number;
  accuracy: number;
  valAccuracy: number;
}

interface ThreeDResearchManifoldProps {
  project: FlagshipProject;
  mlTechnique?: MlTechniqueId;
  dataType?: SyntheticDataType;
  epochPoints?: EpochBarPoint[];
  initialMode?: ThreeDGraphMode;
  heightClass?: string;
}

const _dummyObj = new THREE.Object3D();
const _tmpColor = new THREE.Color();

function seededRand(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function createTextSprite(
  text: string,
  textColor: string,
  bgColor: string,
  borderColor: string,
  maxAnisotropy: number,
  scaleX = 1.9,
  scaleY = 0.5
) {
  const canvas = document.createElement('canvas');
  canvas.width = 640;
  canvas.height = 160;
  const ctx = canvas.getContext('2d');
  if (ctx) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Crisp badge container with high-DPI padding
    const r = 26;
    const w = canvas.width - 16;
    const h = canvas.height - 16;
    const x = 8;
    const y = 8;
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();

    ctx.fillStyle = bgColor;
    ctx.fill();
    ctx.lineWidth = 5;
    ctx.strokeStyle = borderColor;
    ctx.stroke();

    ctx.font = '700 44px "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = textColor;
    ctx.fillText(text, canvas.width / 2, canvas.height / 2 + 2);
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = maxAnisotropy;
  texture.needsUpdate = true;

  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthWrite: false
  });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(scaleX, scaleY, 1);
  return { sprite, texture, material };
}

export default function ThreeDResearchManifold({
  project,
  mlTechnique = 'multimodal_transformer',
  dataType = 'multimodal',
  epochPoints,
  initialMode = 'loss_surface',
  heightClass = 'h-[470px]'
}: ThreeDResearchManifoldProps) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const [mode, setMode] = useState<ThreeDGraphMode>(initialMode);
  const [dockOpen, setDockOpen] = useState(true);
  const cameraPresetRef = useRef<(preset: 'iso' | 'top' | 'front') => void>(() => {});
  const rebuildRef = useRef<() => void>(() => {});

  const points: EpochBarPoint[] =
    epochPoints && epochPoints.length > 0
      ? epochPoints
      : project.epochHistory.map((h) => ({
          epoch: h.epoch,
          loss: h.loss,
          accuracy: h.accuracy,
          valAccuracy: h.valAccuracy ?? Math.max(0.4, h.accuracy - 0.02)
        }));

  const stateRef = useRef({ mode, project, mlTechnique, dataType, points });
  stateRef.current = { mode, project, mlTechnique, dataType, points };

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const isSoftwareGl = (() => {
      const probe = document.createElement('canvas').getContext('webgl2');
      if (!probe) return true;
      const ext = probe.getExtension('WEBGL_debug_renderer_info');
      const param = ext ? ext.UNMASKED_RENDERER_WEBGL : probe.RENDERER;
      const gpu = String(probe.getParameter(param) || '');
      probe.getExtension('WEBGL_lose_context')?.loseContext();
      return /SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(gpu);
    })();

    const width = Math.max(container.clientWidth, 320);
    const height = Math.max(container.clientHeight, 300);

    const renderer = new THREE.WebGLRenderer({
      antialias: !isSoftwareGl,
      powerPreference: 'high-performance'
    });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.08;
    renderer.setPixelRatio(isSoftwareGl ? 1 : Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = !isSoftwareGl;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    const maxAnisotropy = renderer.capabilities.getMaxAnisotropy();

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf1f5f9);
    scene.fog = new THREE.FogExp2(0xf1f5f9, 0.022);

    const pmremGenerator = new THREE.PMREMGenerator(renderer);
    const roomEnv = new RoomEnvironment();
    const envTex = pmremGenerator.fromScene(roomEnv, 0.04).texture;
    scene.environment = envTex;
    roomEnv.dispose();
    pmremGenerator.dispose();

    const camera = new THREE.PerspectiveCamera(44, width / height, 0.1, 120);
    camera.position.set(10.2, 7.4, 10.8);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = !isSoftwareGl;
    controls.dampingFactor = 0.07;
    controls.target.set(0, 1.3, 0);
    controls.minDistance = 5.0;
    controls.maxDistance = 26;
    controls.minPolarAngle = 0.05;
    controls.maxPolarAngle = Math.PI - 0.05;

    // 3-Point Warm/Cool Studio Rig
    const hemiLight = new THREE.HemisphereLight(0xdbeafe, 0x94a3b8, 0.85);
    scene.add(hemiLight);

    const keyLight = new THREE.DirectionalLight(0xfffbeb, 1.85);
    keyLight.position.set(7, 12, 6);
    keyLight.castShadow = !isSoftwareGl;
    keyLight.shadow.mapSize.set(1024, 1024);
    keyLight.shadow.camera.left = -8;
    keyLight.shadow.camera.right = 8;
    keyLight.shadow.camera.top = 8;
    keyLight.shadow.camera.bottom = -8;
    keyLight.shadow.bias = -0.0005;
    keyLight.shadow.normalBias = 0.02;
    scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0x818cf8, 0.75);
    rimLight.position.set(-8, 6, -7);
    scene.add(rimLight);

    // Grounded Plinth + Reference Coordinate Grid
    const plinthGeo = new THREE.BoxGeometry(11.6, 0.24, 11.6);
    const plinthMat = new THREE.MeshStandardMaterial({
      color: 0xe2e8f0,
      roughness: 0.72,
      metalness: 0.08
    });
    const plinth = new THREE.Mesh(plinthGeo, plinthMat);
    plinth.position.set(0, -0.14, 0);
    plinth.receiveShadow = true;
    scene.add(plinth);

    const gridHelper = new THREE.GridHelper(11.2, 14, 0x94a3b8, 0xcbd5e1);
    gridHelper.position.y = 0.01;
    scene.add(gridHelper);

    const graphGroup = new THREE.Group();
    scene.add(graphGroup);

    const disposables: { dispose: () => void }[] = [];
    const clearGraphGroup = () => {
      while (graphGroup.children.length > 0) {
        const child = graphGroup.children[0];
        graphGroup.remove(child);
      }
      while (disposables.length > 0) {
        disposables.pop()?.dispose();
      }
    };

    const addLabel = (
      text: string,
      pos: [number, number, number],
      textColor = '#0f172a',
      bgColor = 'rgba(255,255,255,0.95)',
      borderColor = '#94a3b8',
      sx = 1.65,
      sy = 0.44
    ) => {
      const { sprite, texture, material } = createTextSprite(
        text,
        textColor,
        bgColor,
        borderColor,
        maxAnisotropy,
        sx,
        sy
      );
      sprite.position.set(pos[0], pos[1], pos[2]);
      sprite.renderOrder = 10;
      graphGroup.add(sprite);
      disposables.push(texture, material);
    };

    const buildCoordinateFrame = (
      xAxisTitle: string,
      yAxisTitle: string,
      zAxisTitle: string,
      xTicks: string[],
      yTicks: string[],
      zTicks: string[]
    ) => {
      const ox = -5.1;
      const oy = 0.04;
      const oz = -5.1;
      const xLen = 10.2;
      const yLen = 4.4;
      const zLen = 10.2;

      const addAxisLine = (p1: [number, number, number], p2: [number, number, number], hex: number) => {
        const geo = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(...p1),
          new THREE.Vector3(...p2)
        ]);
        const mat = new THREE.LineBasicMaterial({ color: hex });
        graphGroup.add(new THREE.Line(geo, mat));
        disposables.push(geo, mat);
      };

      // X (Red/Rose), Y (Emerald), Z (Indigo/Blue) outer frame axes (outside center geometry)
      addAxisLine([ox, oy, oz], [ox + xLen, oy, oz], 0xbe123c);
      addAxisLine([ox, oy, oz], [ox, oy + yLen, oz], 0x059669);
      addAxisLine([ox, oy, oz], [ox, oy, oz + zLen], 0x1d4ed8);

      // Axis Arrows (small cones at tips)
      const coneGeo = new THREE.ConeGeometry(0.14, 0.36, 12);
      const coneMatX = new THREE.MeshStandardMaterial({ color: 0xbe123c, roughness: 0.4 });
      const coneX = new THREE.Mesh(coneGeo, coneMatX);
      coneX.position.set(ox + xLen + 0.12, oy, oz);
      coneX.rotation.z = -Math.PI / 2;
      graphGroup.add(coneX);

      const coneMatY = new THREE.MeshStandardMaterial({ color: 0x059669, roughness: 0.4 });
      const coneY = new THREE.Mesh(coneGeo, coneMatY);
      coneY.position.set(ox, oy + yLen + 0.12, oz);
      graphGroup.add(coneY);

      const coneMatZ = new THREE.MeshStandardMaterial({ color: 0x1d4ed8, roughness: 0.4 });
      const coneZ = new THREE.Mesh(coneGeo, coneMatZ);
      coneZ.position.set(ox, oy, oz + zLen + 0.12);
      coneZ.rotation.x = Math.PI / 2;
      graphGroup.add(coneZ);

      disposables.push(coneGeo, coneMatX, coneMatY, coneMatZ);

      // Axis endpoint title sprites placed clearly outside data bounds
      addLabel(xAxisTitle, [ox + xLen + 0.9, oy + 0.36, oz], '#9f1239', '#fff1f2', '#fda4af', 2.6, 0.58);
      addLabel(yAxisTitle, [ox, oy + yLen + 0.72, oz], '#065f46', '#ecfdf5', '#6ee7b7', 2.6, 0.58);
      addLabel(zAxisTitle, [ox, oy + 0.36, oz + zLen + 0.9], '#1e3a8a', '#eff6ff', '#93c5fd', 2.6, 0.58);
      addLabel('(0, 0, 0)', [ox - 0.6, oy + 0.28, oz - 0.6], '#334155', '#f8fafc', '#94a3b8', 1.75, 0.46);

      // Spaced coordinate tick sprites along the front/left outer rims (well clear of geometry)
      xTicks.forEach((lbl, idx) => {
        const px = -4.2 + idx * (8.4 / Math.max(1, xTicks.length - 1));
        addLabel(lbl, [px, oy + 0.24, 5.45], '#334155', 'rgba(255,255,255,0.95)', '#cbd5e1', 1.32, 0.42);
      });
      yTicks.forEach((lbl, idx) => {
        const py = 0.9 + idx * 1.15;
        addLabel(lbl, [ox - 0.78, py, oz], '#065f46', 'rgba(255,255,255,0.95)', '#6ee7b7', 1.32, 0.42);
      });
      zTicks.forEach((lbl, idx) => {
        const pz = -4.2 + idx * (8.4 / Math.max(1, zTicks.length - 1));
        addLabel(lbl, [-5.45, oy + 0.24, pz], '#1e3a8a', 'rgba(255,255,255,0.95)', '#93c5fd', 1.32, 0.42);
      });
    };

    const buildGraph = () => {
      clearGraphGroup();
      const { mode: activeMode, project: proj, mlTechnique: mlId, dataType: dt, points: pts } =
        stateRef.current;
      const hash =
        proj.id.split('').reduce((a, c) => a + c.charCodeAt(0), 0) + mlId.length * 19 + dt.length * 7;
      const rand = seededRand(hash);

      if (activeMode === 'loss_surface') {
        buildCoordinateFrame(
          'X: Weight θ₁',
          'Y: Loss L(θ)',
          'Z: Bias θ₂',
          ['-4.0', '0.0', '+4.0'],
          ['1.0', '2.0', '3.0'],
          ['-4.0', '0.0', '+4.0']
        );

        const segs = 32;
        const size = 8.4;
        const planeGeo = new THREE.PlaneGeometry(size, size, segs, segs);
        planeGeo.rotateX(-Math.PI / 2);

        const posAttr = planeGeo.attributes.position as THREE.BufferAttribute;
        const colors: number[] = [];
        for (let i = 0; i < posAttr.count; i++) {
          const x = posAttr.getX(i);
          const z = posAttr.getZ(i);
          const rSq = (x * x + z * z) * 0.095;
          const ripple = Math.sin(x * 1.15) * Math.cos(z * 1.15) * 0.48;
          const basin = 2.85 * (1 - Math.exp(-rSq)) + ripple + 0.35;
          const h = Math.max(0.18, basin);
          posAttr.setY(i, h);

          // 7-stop scientific Turbo/Spectral gradient: Indigo -> Cyan -> Emerald -> Lime -> Gold -> Orange -> Crimson
          const t = Math.max(0, Math.min(1, h / 3.4));
          if (t < 0.16) {
            const f = t / 0.16;
            _tmpColor.setRGB(0.28 + (0.02 - 0.28) * f, 0.24 + (0.71 - 0.24) * f, 0.88 + (0.83 - 0.88) * f);
          } else if (t < 0.35) {
            const f = (t - 0.16) / 0.19;
            _tmpColor.setRGB(0.02 + (0.06 - 0.02) * f, 0.71 + (0.73 - 0.71) * f, 0.83 + (0.51 - 0.83) * f);
          } else if (t < 0.52) {
            const f = (t - 0.35) / 0.17;
            _tmpColor.setRGB(0.06 + (0.52 - 0.06) * f, 0.73 + (0.80 - 0.73) * f, 0.51 + (0.09 - 0.51) * f);
          } else if (t < 0.68) {
            const f = (t - 0.52) / 0.16;
            _tmpColor.setRGB(0.52 + (0.96 - 0.52) * f, 0.80 + (0.68 - 0.80) * f, 0.09 + (0.04 - 0.09) * f);
          } else if (t < 0.84) {
            const f = (t - 0.68) / 0.16;
            _tmpColor.setRGB(0.96 + (0.98 - 0.96) * f, 0.68 + (0.42 - 0.68) * f, 0.04 + (0.09 - 0.04) * f);
          } else {
            const f = (t - 0.84) / 0.16;
            _tmpColor.setRGB(0.98 + (0.90 - 0.98) * f, 0.42 + (0.11 - 0.42) * f, 0.09 + (0.28 - 0.09) * f);
          }
          colors.push(_tmpColor.r, _tmpColor.g, _tmpColor.b);
        }
        planeGeo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        planeGeo.computeVertexNormals();

        const surfaceMat = new THREE.MeshStandardMaterial({
          vertexColors: true,
          roughness: 0.28,
          metalness: 0.22,
          side: THREE.DoubleSide
        });
        const surfaceMesh = new THREE.Mesh(planeGeo, surfaceMat);
        surfaceMesh.castShadow = !isSoftwareGl;
        surfaceMesh.receiveShadow = true;
        graphGroup.add(surfaceMesh);

        // Topographic wireframe overlay highlighting contours and saddles
        const wireMat = new THREE.MeshBasicMaterial({
          color: 0xffffff,
          wireframe: true,
          transparent: true,
          opacity: 0.22
        });
        const wireMesh = new THREE.Mesh(planeGeo, wireMat);
        wireMesh.position.y = 0.005;
        graphGroup.add(wireMesh);
        disposables.push(planeGeo, surfaceMat, wireMat);

        const curvePts = pts.map((p, idx) => {
          const progress = idx / Math.max(1, pts.length - 1);
          const angle = progress * Math.PI * 1.45;
          const radius = (1 - progress) * 3.4 + 0.15;
          const px = Math.cos(angle) * radius;
          const pz = Math.sin(angle) * radius;
          const py = Math.max(0.45, p.loss * 1.1 + 0.45);
          return new THREE.Vector3(px, py, pz);
        });
        if (curvePts.length >= 2) {
          const curve = new THREE.CatmullRomCurve3(curvePts);
          const tubeGeo = new THREE.TubeGeometry(curve, 54, 0.11, 12, false);
          // Glowing golden iridescent descent trajectory tube
          const tubeMat = new THREE.MeshStandardMaterial({
            color: 0xfbbf24,
            roughness: 0.18,
            metalness: 0.72,
            emissive: 0xb45309,
            emissiveIntensity: 0.35
          });
          const tubeMesh = new THREE.Mesh(tubeGeo, tubeMat);
          tubeMesh.castShadow = !isSoftwareGl;
          graphGroup.add(tubeMesh);
          disposables.push(tubeGeo, tubeMat);

          // Milestone waypoint epoch beads with dynamic chromatic ramp (Crimson -> Gold -> Cyan -> Emerald)
          const beadGeo = new THREE.SphereGeometry(0.12, 16, 16);
          const beadColors = [0xf43f5e, 0xf97316, 0xfbbf24, 0x06b6d4, 0x10b981];
          curvePts.forEach((pt, bIdx) => {
            if (bIdx % Math.max(1, Math.floor(curvePts.length / 5)) === 0 || bIdx === curvePts.length - 1) {
              const bColor = beadColors[Math.min(beadColors.length - 1, Math.floor((bIdx / curvePts.length) * beadColors.length))];
              const beadMat = new THREE.MeshStandardMaterial({
                color: bColor,
                roughness: 0.15,
                metalness: 0.5,
                emissive: bColor,
                emissiveIntensity: 0.2
              });
              const bead = new THREE.Mesh(beadGeo, beadMat);
              bead.position.copy(pt);
              graphGroup.add(bead);
              disposables.push(beadMat);
            }
          });
          disposables.push(beadGeo);

          // Converged Global Minimum with radiant emerald core and horizontal neon cyan halo ring
          const endPt = curvePts[curvePts.length - 1];
          const orbGeo = new THREE.SphereGeometry(0.26, 24, 24);
          const orbMat = new THREE.MeshStandardMaterial({
            color: 0x10b981,
            roughness: 0.15,
            metalness: 0.5,
            emissive: 0x059669,
            emissiveIntensity: 0.45
          });
          const orb = new THREE.Mesh(orbGeo, orbMat);
          orb.position.copy(endPt);
          graphGroup.add(orb);

          const haloGeo = new THREE.TorusGeometry(0.52, 0.04, 16, 40);
          haloGeo.rotateX(Math.PI / 2);
          const haloMat = new THREE.MeshBasicMaterial({
            color: 0x06b6d4,
            transparent: true,
            opacity: 0.85
          });
          const halo = new THREE.Mesh(haloGeo, haloMat);
          halo.position.copy(endPt);
          graphGroup.add(halo);
          disposables.push(orbGeo, orbMat, haloGeo, haloMat);

          const minLoss = pts[pts.length - 1]?.loss.toFixed(3) ?? '0.05';
          addLabel(
            `★ Global Min: θ* (Loss: ${minLoss})`,
            [endPt.x, endPt.y + 1.25, endPt.z],
            '#065f46',
            '#ecfdf5',
            '#10b981',
            2.75,
            0.52
          );
        }
      } else if (activeMode === 'val_acc_pillars') {
        const count = Math.min(pts.length, 12);
        buildCoordinateFrame(
          'X: Epoch',
          'Y: Accuracy %',
          'Z: Track',
          ['E1', `E${Math.ceil(count / 2)}`, `E${count}`],
          ['33%', '66%', '100%'],
          ['Val Acc (Front)', 'Train Acc (Mid)', 'Baseline (Back)']
        );

        const colSpacing = 8.0 / Math.max(1, count);
        const barGeo = new THREE.BoxGeometry(colSpacing * 0.58, 1, 0.95);
        barGeo.translate(0, 0.5, 0);
        const mat = new THREE.MeshStandardMaterial({ roughness: 0.22, metalness: 0.25 });
        const rows = 3;
        const inst = new THREE.InstancedMesh(barGeo, mat, count * rows);
        inst.castShadow = !isSoftwareGl;
        inst.receiveShadow = true;

        let idx = 0;
        for (let r = 0; r < rows; r++) {
          for (let i = 0; i < count; i++) {
            const pt = pts[i];
            const metricVal =
              r === 0 ? pt.valAccuracy : r === 1 ? pt.accuracy : Math.max(0.3, pt.valAccuracy - 0.12);
            const h = Math.max(0.25, metricVal * 4.0);
            const x = -4.0 + (i + 0.5) * colSpacing;
            const z = (r - 1) * 1.9;
            _dummyObj.position.set(x, 0.02, z);
            _dummyObj.scale.set(1, h, 1);
            _dummyObj.updateMatrix();
            inst.setMatrixAt(idx, _dummyObj.matrix);

            if (r === 0) {
              // Front Track (Validation Accuracy): Rich multi-tier performance gradient
              if (pt.valAccuracy >= 0.96) {
                _tmpColor.setHex(0x00f59b); // Radiant Spring Mint (Exceptional SLA)
              } else if (pt.valAccuracy >= 0.93) {
                _tmpColor.setHex(0x10b981); // Emerald (Target SLA met)
              } else if (pt.valAccuracy >= 0.85) {
                _tmpColor.setHex(0x06b6d4); // Cyan / Turquoise
              } else if (pt.valAccuracy >= 0.75) {
                _tmpColor.setHex(0xf59e0b); // Golden Amber
              } else {
                _tmpColor.setHex(0xf43f5e); // Hot Coral / Crimson
              }
            } else if (r === 1) {
              // Middle Track (Training Accuracy): Sapphire -> Electric Indigo -> Violet
              const progress = i / Math.max(1, count - 1);
              if (progress < 0.4) {
                _tmpColor.setHex(0x2563eb); // Royal Sapphire
              } else if (progress < 0.75) {
                _tmpColor.setHex(0x6366f1); // Electric Indigo
              } else {
                _tmpColor.setHex(0x8b5cf6); // Vibrant Purple / Violet
              }
            } else {
              // Back Track (Baseline Prior): Warm Bronze / Sunset Amber spectrum instead of dull grey
              const progress = i / Math.max(1, count - 1);
              _tmpColor.setHex(progress > 0.5 ? 0xea580c : 0xd97706);
            }
            inst.setColorAt(idx, _tmpColor);
            idx++;
          }
        }
        inst.instanceMatrix.needsUpdate = true;
        if (inst.instanceColor) inst.instanceColor.needsUpdate = true;
        inst.computeBoundingSphere();
        graphGroup.add(inst);
        disposables.push(barGeo, mat, inst);

        // Track Identity floating badges along the right edge of each track
        addLabel('Val Track (SLA ≥95%)', [4.6, 1.8, -1.9], '#065f46', '#ecfdf5', '#10b981', 2.3, 0.44);
        addLabel('Train Track (Loss Min)', [4.6, 1.8, 0.0], '#1e1b4b', '#eef2ff', '#6366f1', 2.3, 0.44);
        addLabel('Baseline Prior', [4.6, 1.8, 1.9], '#78350f', '#fef3c7', '#d97706', 2.1, 0.44);

        // Peak Val pillar callout cleanly above the tallest front bar
        const lastPt = pts[count - 1];
        if (lastPt) {
          const peakX = -4.0 + (count - 0.5) * colSpacing;
          const peakH = Math.max(0.25, lastPt.valAccuracy * 4.0);
          addLabel(
            `★ Peak Val Acc: ${(lastPt.valAccuracy * 100).toFixed(1)}% (E${lastPt.epoch})`,
            [peakX, peakH + 0.65, -1.9],
            '#065f46',
            '#ecfdf5',
            '#10b981',
            2.65,
            0.52
          );
        }
      } else if (activeMode === 'k9_clusters') {
        buildCoordinateFrame(
          'X: Latent Z₁',
          'Y: Latent Z₂',
          'Z: Latent Z₃',
          ['-4.0', '0.0', '+4.0'],
          ['1.0', '2.0', '3.0'],
          ['-4.0', '0.0', '+4.0']
        );

        // 6 distinct vibrant cluster color families with bright orbital halo rings
        const centroids = [
          { label: 'C1: Core High-Density', pos: new THREE.Vector3(-2.8, 1.4, -2.4), color: '#065f46', hexColor: 0x10b981, ringColor: 0x34d399 },
          { label: 'C2: Outlier Frontier', pos: new THREE.Vector3(2.8, 1.9, -2.2), color: '#9f1239', hexColor: 0xf43f5e, ringColor: 0xfb7185 },
          { label: 'C3: Solar High-Var', pos: new THREE.Vector3(2.5, 1.2, 2.6), color: '#b45309', hexColor: 0xfbbf24, ringColor: 0xfde047 },
          { label: 'C4: Latent Azure', pos: new THREE.Vector3(-2.6, 2.6, 2.4), color: '#0369a1', hexColor: 0x0ea5e9, ringColor: 0x7dd3fc },
          { label: 'C5: Violet Manifold', pos: new THREE.Vector3(0.0, 3.1, -1.2), color: '#6b21a8', hexColor: 0xa855f7, ringColor: 0xd8b4fe },
          { label: 'C6: Flame Transition', pos: new THREE.Vector3(1.8, 2.7, 0.4), color: '#c2410c', hexColor: 0xf97316, ringColor: 0xfdba74 }
        ];

        const nPts = 84;
        const sphereGeo = new THREE.SphereGeometry(0.16, 16, 16);
        const ptsMat = new THREE.MeshStandardMaterial({ roughness: 0.22, metalness: 0.35 });
        const instPts = new THREE.InstancedMesh(sphereGeo, ptsMat, nPts);
        instPts.castShadow = !isSoftwareGl;

        const positions: { pos: THREE.Vector3; clusterIdx: number }[] = [];
        for (let i = 0; i < nPts; i++) {
          const cIdx = i % centroids.length;
          const c = centroids[cIdx].pos;
          const p = new THREE.Vector3(
            c.x + (rand() - 0.5) * 2.3,
            Math.max(0.28, c.y + (rand() - 0.5) * 1.8),
            c.z + (rand() - 0.5) * 2.3
          );
          positions.push({ pos: p, clusterIdx: cIdx });
          _dummyObj.position.copy(p);
          _dummyObj.scale.set(1, 1, 1);
          _dummyObj.updateMatrix();
          instPts.setMatrixAt(i, _dummyObj.matrix);
          _tmpColor.setHex(centroids[cIdx].hexColor);
          instPts.setColorAt(i, _tmpColor);
        }
        instPts.instanceMatrix.needsUpdate = true;
        if (instPts.instanceColor) instPts.instanceColor.needsUpdate = true;
        instPts.computeBoundingSphere();
        graphGroup.add(instPts);
        disposables.push(sphereGeo, ptsMat, instPts);

        // Centroid Spheres + Colored Orbital Halo Rings
        const centroidGeo = new THREE.SphereGeometry(0.3, 20, 20);
        const ringGeo = new THREE.TorusGeometry(0.72, 0.045, 14, 36);
        ringGeo.rotateX(Math.PI / 2.6);

        centroids.forEach((c) => {
          const cMat = new THREE.MeshStandardMaterial({
            color: c.hexColor,
            roughness: 0.18,
            metalness: 0.55,
            emissive: c.hexColor,
            emissiveIntensity: 0.28
          });
          const cMesh = new THREE.Mesh(centroidGeo, cMat);
          cMesh.position.copy(c.pos);
          graphGroup.add(cMesh);

          const rMat = new THREE.MeshBasicMaterial({
            color: c.ringColor,
            transparent: true,
            opacity: 0.85
          });
          const rMesh = new THREE.Mesh(ringGeo, rMat);
          rMesh.position.copy(c.pos);
          graphGroup.add(rMesh);

          disposables.push(cMat, rMat);
          addLabel(c.label, [c.pos.x, c.pos.y + 1.25, c.pos.z], c.color, 'rgba(255,255,255,0.96)', c.color, 2.5, 0.44);
        });
        disposables.push(centroidGeo, ringGeo);

        // Radiant Gold Diamond Query Node q(0.2, 1.8, 0.1)
        const queryPos = new THREE.Vector3(0.2, 1.8, 0.1);
        const qGeo = new THREE.OctahedronGeometry(0.38, 0);
        const qMat = new THREE.MeshStandardMaterial({
          color: 0xfbbf24,
          roughness: 0.15,
          metalness: 0.75,
          emissive: 0xd97706,
          emissiveIntensity: 0.45
        });
        const qMesh = new THREE.Mesh(qGeo, qMat);
        qMesh.position.copy(queryPos);
        graphGroup.add(qMesh);

        // Electric Indigo target halo ring around query point
        const qHaloGeo = new THREE.TorusGeometry(0.65, 0.04, 16, 36);
        qHaloGeo.rotateX(Math.PI / 2);
        const qHaloMat = new THREE.MeshBasicMaterial({ color: 0x6366f1, transparent: true, opacity: 0.85 });
        const qHaloMesh = new THREE.Mesh(qHaloGeo, qHaloMat);
        qHaloMesh.position.copy(queryPos);
        graphGroup.add(qHaloMesh);
        disposables.push(qGeo, qMat, qHaloGeo, qHaloMat);

        addLabel('★ Query Node q(0.2, 1.8, 0.1) · k=9', [queryPos.x, queryPos.y + 1.05, queryPos.z], '#1e1b4b', '#eef2ff', '#4f46e5', 2.85, 0.52);

        // Calculate 9 Nearest Neighbors
        const k9Neighbors = [...positions]
          .sort((a, b) => a.pos.distanceToSquared(queryPos) - b.pos.distanceToSquared(queryPos))
          .slice(0, 9);

        // Multi-color k=9 rays connecting query point to neighbors
        const neighborRingGeo = new THREE.TorusGeometry(0.28, 0.035, 12, 24);
        neighborRingGeo.rotateX(Math.PI / 2);

        k9Neighbors.forEach((n) => {
          const cData = centroids[n.clusterIdx];
          const lineGeo = new THREE.BufferGeometry().setFromPoints([queryPos, n.pos]);
          const lineMat = new THREE.LineBasicMaterial({
            color: cData.hexColor,
            linewidth: 2,
            transparent: true,
            opacity: 0.92
          });
          graphGroup.add(new THREE.Line(lineGeo, lineMat));

          // Neighbor Highlight Ring
          const nRingMat = new THREE.MeshBasicMaterial({ color: cData.ringColor, transparent: true, opacity: 0.95 });
          const nRingMesh = new THREE.Mesh(neighborRingGeo, nRingMat);
          nRingMesh.position.copy(n.pos);
          graphGroup.add(nRingMesh);

          disposables.push(lineGeo, lineMat, nRingMat);
        });
        disposables.push(neighborRingGeo);
      } else {
        // 3D Tensor Voxels
        buildCoordinateFrame(
          'X: Batch [0..5]',
          'Y: Layer [0..3]',
          'Z: Seq [0..5]',
          ['B=0', 'B=2', 'B=5'],
          ['L=0 (Emb)', 'L=1 (Attn)', 'L=3 (Head)'],
          ['S=0', 'S=2', 'S=5']
        );

        const nx = 6;
        const ny = 4;
        const nz = 6;
        const total = nx * ny * nz;
        const voxGeo = new THREE.BoxGeometry(0.54, 0.54, 0.54);
        const voxMat = new THREE.MeshStandardMaterial({ roughness: 0.22, metalness: 0.28 });
        const instVox = new THREE.InstancedMesh(voxGeo, voxMat, total);
        instVox.castShadow = !isSoftwareGl;

        let maxAct = -1;
        let maxPos = new THREE.Vector3();
        let idx = 0;

        for (let y = 0; y < ny; y++) {
          for (let x = 0; x < nx; x++) {
            for (let z = 0; z < nz; z++) {
              const act = Math.abs(Math.sin(x * 0.72 + y * 0.88 + z * 0.52 + rand() * 0.35));
              const s = 0.38 + act * 0.65;
              const vx = (x - 2.5) * 1.1;
              const vy = 0.48 + y * 0.95;
              const vz = (z - 2.5) * 1.1;
              _dummyObj.position.set(vx, vy, vz);
              _dummyObj.scale.set(s, s, s);
              _dummyObj.updateMatrix();
              instVox.setMatrixAt(idx, _dummyObj.matrix);

              if (act > maxAct) {
                maxAct = act;
                maxPos.set(vx, vy, vz);
              }

              // Full-spectrum thermal activation palette: Cosmic Indigo -> Cyan -> Emerald -> Sun Gold -> Hot Crimson
              if (act < 0.22) {
                _tmpColor.setRGB(0.24, 0.20, 0.65); // Cosmic Indigo/Violet
              } else if (act < 0.42) {
                _tmpColor.setRGB(0.04, 0.71, 0.92); // Electric Cyan
              } else if (act < 0.62) {
                _tmpColor.setRGB(0.08, 0.82, 0.42); // Radiant Emerald
              } else if (act < 0.82) {
                _tmpColor.setRGB(0.98, 0.72, 0.08); // Solar Gold/Amber
              } else {
                _tmpColor.setRGB(0.96, 0.18, 0.36); // Hot Flame Crimson
              }

              instVox.setColorAt(idx, _tmpColor);
              idx++;
            }
          }
        }
        instVox.instanceMatrix.needsUpdate = true;
        if (instVox.instanceColor) instVox.instanceColor.needsUpdate = true;
        instVox.computeBoundingSphere();
        graphGroup.add(instVox);
        disposables.push(voxGeo, voxMat, instVox);

        // 4 Layer Accent Slices: Horizontal colored boundary frame for each layer
        const layerInfo = [
          { name: 'L0: Token Emb (B×S×D)', color: 0x06b6d4, textColor: '#0e7490', bg: '#ecfeff', border: '#06b6d4' },
          { name: 'L1: Self-Attn (8-Head)', color: 0x8b5cf6, textColor: '#5b21b6', bg: '#f5f3ff', border: '#8b5cf6' },
          { name: 'L2: FFN SwiGLU (4×D)', color: 0xf59e0b, textColor: '#92400e', bg: '#fffbeb', border: '#f59e0b' },
          { name: 'L3: Output Head Logits', color: 0x10b981, textColor: '#065f46', bg: '#ecfdf5', border: '#10b981' }
        ];

        const sliceGeo = new THREE.EdgesGeometry(new THREE.BoxGeometry(6.6, 0.05, 6.6));
        layerInfo.forEach((info, lIdx) => {
          const vy = 0.48 + lIdx * 0.95;
          const sliceMat = new THREE.LineBasicMaterial({
            color: info.color,
            transparent: true,
            opacity: 0.65
          });
          const sliceMesh = new THREE.LineSegments(sliceGeo, sliceMat);
          sliceMesh.position.set(0, vy, 0);
          graphGroup.add(sliceMesh);
          disposables.push(sliceMat);

          addLabel(info.name, [-3.8, vy + 0.12, 3.4], info.textColor, info.bg, info.border, 2.5, 0.42);
        });
        disposables.push(sliceGeo);

        // Peak Activation Beacon & Label
        const beaconGeo = new THREE.OctahedronGeometry(0.24, 0);
        const beaconMat = new THREE.MeshBasicMaterial({ color: 0xfbbf24 });
        const beacon = new THREE.Mesh(beaconGeo, beaconMat);
        beacon.position.set(maxPos.x, maxPos.y + 0.65, maxPos.z);
        graphGroup.add(beacon);
        disposables.push(beaconGeo, beaconMat);

        addLabel(
          `★ Peak Attn: ${(maxAct).toFixed(3)} (L3·Head 2)`,
          [maxPos.x, maxPos.y + 1.15, maxPos.z],
          '#9f1239',
          '#fff1f2',
          '#f43f5e',
          2.6,
          0.48
        );
      }
    };

    // On-demand coalesced rendering loop (software-WebGL paced safely)
    let frameId = 0;
    let renderBusy = false;
    let renderDirty = false;
    let disposed = false;

    const afterDraw = (cb: () => void) => {
      const gl = renderer.getContext() as WebGL2RenderingContext;
      const sync = gl.fenceSync ? gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0) : null;
      if (!sync) {
        frameId = window.setTimeout(cb, 80);
        return;
      }
      gl.flush();
      const poll = () => {
        if (disposed) return;
        const state = gl.getSyncParameter(sync, gl.SYNC_STATUS);
        if (state === gl.UNSIGNALED) {
          frameId = window.setTimeout(poll, 5);
        } else {
          gl.deleteSync(sync);
          frameId = window.setTimeout(cb, 80);
        }
      };
      frameId = window.setTimeout(poll, 5);
    };

    const requestRender = () => {
      if (disposed) return;
      if (renderBusy) {
        renderDirty = true;
        return;
      }
      renderBusy = true;
      frameId = requestAnimationFrame(() => {
        if (disposed) return;
        renderDirty = false;
        controls.update();
        renderer.render(scene, camera);
        const done = () => {
          renderBusy = false;
          if (renderDirty && !disposed) requestRender();
        };
        if (isSoftwareGl) afterDraw(done);
        else done();
      });
    };

    cameraPresetRef.current = (preset) => {
      if (preset === 'iso') {
        camera.position.set(10.2, 7.4, 10.8);
      } else if (preset === 'top') {
        camera.position.set(0.01, 15.0, 0.01);
      } else {
        camera.position.set(0, 3.8, 13.5);
      }
      controls.target.set(0, 1.3, 0);
      controls.update();
      requestRender();
    };

    rebuildRef.current = () => {
      buildGraph();
      requestRender();
    };

    buildGraph();
    if (!isSoftwareGl) {
      renderer.compile(scene, camera);
    }
    requestRender();

    const handleControlsChange = () => requestRender();
    controls.addEventListener('change', handleControlsChange);

    const handleResize = () => {
      if (!container || disposed) return;
      const w = Math.max(container.clientWidth, 320);
      const h = Math.max(container.clientHeight, 300);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
      requestRender();
    };
    window.addEventListener('resize', handleResize);

    return () => {
      disposed = true;
      cancelAnimationFrame(frameId);
      clearTimeout(frameId);
      window.removeEventListener('resize', handleResize);
      controls.removeEventListener('change', handleControlsChange);
      controls.dispose();
      clearGraphGroup();
      plinthGeo.dispose();
      plinthMat.dispose();
      gridHelper.dispose();
      envTex.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
    };
  }, []);

  useEffect(() => {
    rebuildRef.current();
  }, [mode, project.id, mlTechnique, dataType, points.length]);

  const activeTechName = ML_TECHNIQUE_META[mlTechnique].shortName;
  const activeTypeName = SYNTHETIC_DATA_TYPE_META[dataType].shortLabel;

  const axisSummary =
    mode === 'loss_surface'
      ? 'X: θ₁[-4..4] · Y: Loss[0..3.5] · Z: θ₂[-4..4]'
      : mode === 'val_acc_pillars'
        ? `X: Epoch[1..${points.length}] · Y: Acc[0..100%] · Z: Track`
        : mode === 'k9_clusters'
          ? 'X: Z₁[-4..4] · Y: Z₂[0..3.5] · Z: Z₃[-4..4]'
          : 'X: Batch[0..5] · Y: Layer[0..3] · Z: Seq[0..5]';

  return (
    <div className="rounded-2xl bg-slate-100 border border-slate-300 overflow-hidden shadow-xs relative">
      {/* Isolated WebGL Mount Target */}
      <div className={`w-full ${heightClass} relative bg-slate-100`}>
        <div ref={mountRef} className="w-full h-full" />

        {/* Top-Left Spec Header (Compact, solid background, zero overlap with center manifold) */}
        <div className="pointer-events-none absolute top-3 left-3 bg-white/95 border border-slate-300 rounded-xl px-3.5 py-2 shadow-xs max-w-sm">
          <div className="flex items-center gap-1.5 text-xs font-mono text-indigo-800 font-bold">
            <Box className="w-3.5 h-3.5" />
            <span>Interactive 3D WebGL Manifold (XYZ)</span>
          </div>
          <div className="text-sm font-bold text-slate-900 truncate mt-0.5">{project.title}</div>
          <div className="text-xs font-mono text-slate-700 truncate">
            {activeTechName} · {activeTypeName}
          </div>
        </div>

        {/* Top-Right Compact XYZ Axis Key */}
        <div className="pointer-events-none hidden sm:flex absolute top-3 right-3 bg-white/95 border border-slate-300 rounded-xl px-3.5 py-2 shadow-xs items-center gap-2.5 text-xs font-mono text-slate-800">
          <span className="font-bold text-rose-800">X</span>
          <span className="font-bold text-emerald-800">Y</span>
          <span className="font-bold text-blue-800">Z</span>
          <span className="text-slate-700 font-medium">{axisSummary}</span>
        </div>

        {/* Collapsible Unified Bottom Control Dock */}
        <div className="absolute bottom-3 left-1/2 -translate-x-1/2 w-[94%] max-w-3xl flex flex-col items-center">
          <button
            onClick={() => setDockOpen((v) => !v)}
            className="mb-1 px-3 py-1 rounded-t-lg bg-white border border-slate-300 text-[10.5px] font-mono font-semibold text-slate-700 hover:text-slate-900 flex items-center gap-1 shadow-xs"
          >
            {dockOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronUp className="w-3 h-3" />}
            <span>{dockOpen ? 'Hide 3D Controls' : 'Show 3D Controls'}</span>
          </button>

          {dockOpen && (
            <div className="w-full bg-white/95 border border-slate-300 rounded-xl p-2.5 shadow-md space-y-2">
              {/* Dynamic Chromatic Palette Legend Strip */}
              <div className="flex flex-wrap items-center justify-between gap-2 px-1 text-[10.5px] font-mono border-b border-slate-200 pb-1.5">
                {mode === 'loss_surface' && (
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-bold text-slate-800">Loss Spectrum:</span>
                    <span className="inline-block h-2.5 w-32 rounded-full bg-gradient-to-r from-indigo-700 via-cyan-500 via-emerald-500 via-lime-500 via-amber-400 to-rose-600 shadow-2xs" />
                    <span className="text-indigo-800 font-semibold">Min θ* (Indigo/Cyan)</span>
                    <span className="text-slate-400">→</span>
                    <span className="text-emerald-800 font-semibold">Valley</span>
                    <span className="text-slate-400">→</span>
                    <span className="text-amber-800 font-semibold">Saddle</span>
                    <span className="text-slate-400">→</span>
                    <span className="text-rose-800 font-semibold">Peak (Crimson)</span>
                    <span className="ml-1 inline-flex items-center gap-1 text-amber-700 font-bold bg-amber-50 px-1.5 py-0.5 rounded border border-amber-200">
                      <span className="w-2 h-2 rounded-full bg-amber-400" />
                      <span>Golden Descent Path</span>
                    </span>
                  </div>
                )}

                {mode === 'val_acc_pillars' && (
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-bold text-slate-800">Pillar Tracks:</span>
                    <span className="inline-flex items-center gap-1 text-emerald-800 font-semibold bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-200">
                      <span className="w-2 h-2 rounded-full bg-emerald-500" />
                      <span>Val Acc (Front: Coral &lt;75% · Amber · Cyan · Emerald ≥95%)</span>
                    </span>
                    <span className="inline-flex items-center gap-1 text-indigo-800 font-semibold bg-indigo-50 px-1.5 py-0.5 rounded border border-indigo-200">
                      <span className="w-2 h-2 rounded-full bg-indigo-500" />
                      <span>Train Acc (Mid: Sapphire→Violet)</span>
                    </span>
                    <span className="inline-flex items-center gap-1 text-amber-900 font-semibold bg-amber-50 px-1.5 py-0.5 rounded border border-amber-200">
                      <span className="w-2 h-2 rounded-full bg-orange-600" />
                      <span>Baseline Prior (Back: Bronze Sunset)</span>
                    </span>
                  </div>
                )}

                {mode === 'k9_clusters' && (
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span className="font-bold text-slate-800">6 Clusters:</span>
                    <span className="inline-flex items-center gap-1 text-emerald-900 bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-200 font-semibold">
                      <span className="w-2 h-2 rounded-full bg-emerald-500" /> C1 Emerald
                    </span>
                    <span className="inline-flex items-center gap-1 text-rose-900 bg-rose-50 px-1.5 py-0.5 rounded border border-rose-200 font-semibold">
                      <span className="w-2 h-2 rounded-full bg-rose-500" /> C2 Crimson
                    </span>
                    <span className="inline-flex items-center gap-1 text-amber-900 bg-amber-50 px-1.5 py-0.5 rounded border border-amber-200 font-semibold">
                      <span className="w-2 h-2 rounded-full bg-amber-400" /> C3 Gold
                    </span>
                    <span className="inline-flex items-center gap-1 text-sky-900 bg-sky-50 px-1.5 py-0.5 rounded border border-sky-200 font-semibold">
                      <span className="w-2 h-2 rounded-full bg-sky-500" /> C4 Azure
                    </span>
                    <span className="inline-flex items-center gap-1 text-purple-900 bg-purple-50 px-1.5 py-0.5 rounded border border-purple-200 font-semibold">
                      <span className="w-2 h-2 rounded-full bg-purple-500" /> C5 Violet
                    </span>
                    <span className="inline-flex items-center gap-1 text-orange-900 bg-orange-50 px-1.5 py-0.5 rounded border border-orange-200 font-semibold">
                      <span className="w-2 h-2 rounded-full bg-orange-500" /> C6 Orange
                    </span>
                    <span className="inline-flex items-center gap-1 text-amber-950 font-bold bg-amber-100 px-1.5 py-0.5 rounded border border-amber-300">
                      ★ Query q (Solar Gold) · k=9 Rays
                    </span>
                  </div>
                )}

                {mode === 'tensor_voxels' && (
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-bold text-slate-800">Activation Heat:</span>
                    <span className="inline-block h-2.5 w-28 rounded-full bg-gradient-to-r from-indigo-800 via-cyan-500 via-emerald-500 via-amber-400 to-rose-600 shadow-2xs" />
                    <span className="text-indigo-800 font-semibold">Cosmic Indigo (&lt;0.2)</span>
                    <span className="text-slate-400">→</span>
                    <span className="text-cyan-800 font-semibold">Cyan</span>
                    <span className="text-slate-400">→</span>
                    <span className="text-emerald-800 font-semibold">Emerald</span>
                    <span className="text-slate-400">→</span>
                    <span className="text-amber-800 font-semibold">Gold</span>
                    <span className="text-slate-400">→</span>
                    <span className="text-rose-800 font-semibold">Peak Flame (1.0)</span>
                    <span className="text-slate-600 font-bold">| Slices:</span>
                    <span className="text-cyan-800 font-bold">L0 Emb</span>
                    <span className="text-purple-800 font-bold">L1 Attn</span>
                    <span className="text-amber-800 font-bold">L2 SwiGLU</span>
                    <span className="text-emerald-800 font-bold">L3 Logits</span>
                  </div>
                )}
              </div>

              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex flex-wrap items-center gap-1">
                  {[
                    { id: 'loss_surface', label: '3D Loss Surface' },
                    { id: 'val_acc_pillars', label: '3D Val Acc Pillars' },
                    { id: 'k9_clusters', label: '3D k=9 Clusters' },
                    { id: 'tensor_voxels', label: '3D Tensor Voxels' }
                  ].map((tab) => (
                    <button
                      key={tab.id}
                      onClick={() => setMode(tab.id as ThreeDGraphMode)}
                      className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold transition-colors border ${
                        mode === tab.id
                          ? 'bg-indigo-600 text-white border-indigo-600 shadow-xs'
                          : 'bg-slate-50 text-slate-700 border-slate-300 hover:bg-slate-100'
                      }`}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>

                <div className="flex items-center gap-1">
                  <span className="text-[10px] font-mono text-slate-500 mr-1 flex items-center gap-1">
                    <Compass className="w-3 h-3" /> View:
                  </span>
                  {(['iso', 'top', 'front'] as const).map((cam) => (
                    <button
                      key={cam}
                      onClick={() => cameraPresetRef.current(cam)}
                      className="px-2 py-1 rounded-md text-[10.5px] font-mono font-semibold bg-slate-100 hover:bg-slate-200 text-slate-800 border border-slate-300 uppercase"
                    >
                      {cam}
                    </button>
                  ))}
                  <button
                    onClick={() => cameraPresetRef.current('iso')}
                    title="Reset camera"
                    className="p-1 rounded-md bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-300"
                  >
                    <RotateCcw className="w-3 h-3" />
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
