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
  scaleX = 1.65,
  scaleY = 0.44
) {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 136;
  const ctx = canvas.getContext('2d');
  if (ctx) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Crisp pill container with clear padding
    const r = 24;
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
    ctx.lineWidth = 4;
    ctx.strokeStyle = borderColor;
    ctx.stroke();

    ctx.font = 'bold 34px ui-monospace, SFMono-Regular, Menlo, monospace';
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
  heightClass = 'h-[420px]'
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
      addLabel(xAxisTitle, [ox + xLen + 0.85, oy + 0.32, oz], '#9f1239', '#fff1f2', '#fda4af', 2.25, 0.5);
      addLabel(yAxisTitle, [ox, oy + yLen + 0.65, oz], '#065f46', '#ecfdf5', '#6ee7b7', 2.25, 0.5);
      addLabel(zAxisTitle, [ox, oy + 0.32, oz + zLen + 0.85], '#1e3a8a', '#eff6ff', '#93c5fd', 2.25, 0.5);
      addLabel('(0, 0, 0)', [ox - 0.55, oy + 0.25, oz - 0.55], '#334155', '#f8fafc', '#94a3b8', 1.5, 0.4);

      // Spaced coordinate tick sprites along the front/left outer rims (well clear of geometry)
      xTicks.forEach((lbl, idx) => {
        const px = -4.2 + idx * (8.4 / Math.max(1, xTicks.length - 1));
        addLabel(lbl, [px, oy + 0.22, 5.45], '#475569', 'rgba(255,255,255,0.92)', '#cbd5e1', 1.1, 0.36);
      });
      yTicks.forEach((lbl, idx) => {
        const py = 0.9 + idx * 1.15;
        addLabel(lbl, [ox - 0.72, py, oz], '#065f46', 'rgba(255,255,255,0.92)', '#a7f3d0', 1.1, 0.36);
      });
      zTicks.forEach((lbl, idx) => {
        const pz = -4.2 + idx * (8.4 / Math.max(1, zTicks.length - 1));
        addLabel(lbl, [-5.45, oy + 0.22, pz], '#1e40af', 'rgba(255,255,255,0.92)', '#bfdbfe', 1.1, 0.36);
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

        const segs = 28;
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

          const t = Math.min(1, h / 3.4);
          _tmpColor.setHSL(0.46 - t * 0.42, 0.72, 0.45 + (1 - t) * 0.12);
          colors.push(_tmpColor.r, _tmpColor.g, _tmpColor.b);
        }
        planeGeo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        planeGeo.computeVertexNormals();

        const surfaceMat = new THREE.MeshStandardMaterial({
          vertexColors: true,
          roughness: 0.35,
          metalness: 0.15,
          side: THREE.DoubleSide
        });
        const surfaceMesh = new THREE.Mesh(planeGeo, surfaceMat);
        surfaceMesh.castShadow = !isSoftwareGl;
        surfaceMesh.receiveShadow = true;
        graphGroup.add(surfaceMesh);
        disposables.push(planeGeo, surfaceMat);

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
          const tubeGeo = new THREE.TubeGeometry(curve, 48, 0.09, 10, false);
          const tubeMat = new THREE.MeshStandardMaterial({
            color: 0x1e1b4b,
            roughness: 0.25,
            metalness: 0.55
          });
          const tubeMesh = new THREE.Mesh(tubeGeo, tubeMat);
          graphGroup.add(tubeMesh);
          disposables.push(tubeGeo, tubeMat);

          const endPt = curvePts[curvePts.length - 1];
          const orbGeo = new THREE.SphereGeometry(0.22, 20, 20);
          const orbMat = new THREE.MeshStandardMaterial({
            color: 0x10b981,
            roughness: 0.2,
            metalness: 0.4
          });
          const orb = new THREE.Mesh(orbGeo, orbMat);
          orb.position.copy(endPt);
          graphGroup.add(orb);
          disposables.push(orbGeo, orbMat);

          const minLoss = pts[pts.length - 1]?.loss.toFixed(3) ?? '0.05';
          addLabel(
            `Min (${endPt.x.toFixed(1)}, ${minLoss}, ${endPt.z.toFixed(1)})`,
            [endPt.x, endPt.y + 1.15, endPt.z],
            '#065f46',
            '#ecfdf5',
            '#10b981',
            2.4,
            0.46
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
          ['Val', 'Train', 'Base']
        );

        const colSpacing = 8.0 / Math.max(1, count);
        const barGeo = new THREE.BoxGeometry(colSpacing * 0.58, 1, 0.95);
        barGeo.translate(0, 0.5, 0);
        const mat = new THREE.MeshStandardMaterial({ roughness: 0.28, metalness: 0.18 });
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
              _tmpColor.setHex(pt.valAccuracy >= 0.93 ? 0x059669 : 0x4f46e5);
            } else if (r === 1) {
              _tmpColor.setHex(0x1d4ed8);
            } else {
              _tmpColor.setHex(0x94a3b8);
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

        // Peak Val pillar callout cleanly above the tallest front bar
        const lastPt = pts[count - 1];
        if (lastPt) {
          const peakX = -4.0 + (count - 0.5) * colSpacing;
          const peakH = Math.max(0.25, lastPt.valAccuracy * 4.0);
          addLabel(
            `E${lastPt.epoch} Val: ${(lastPt.valAccuracy * 100).toFixed(1)}%`,
            [peakX, peakH + 0.55, -1.9],
            '#065f46',
            '#ecfdf5',
            '#059669',
            2.15,
            0.44
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

        const centroids = [
          { label: 'C1 (-2.6, 1.4, -2.2)', pos: new THREE.Vector3(-2.6, 1.4, -2.2), color: '#065f46' },
          { label: 'C2 (+2.6, 1.8, -2.0)', pos: new THREE.Vector3(2.6, 1.8, -2.0), color: '#9f1239' },
          { label: 'C3 (+2.4, 1.2, +2.5)', pos: new THREE.Vector3(2.4, 1.2, 2.5), color: '#b45309' },
          { label: 'C4 (-2.4, 2.5, +2.4)', pos: new THREE.Vector3(-2.4, 2.5, 2.4), color: '#1e3a8a' }
        ];
        const clusterColors = [0x059669, 0xbe123c, 0xd97706, 0x2563eb];
        const nPts = 64;
        const sphereGeo = new THREE.SphereGeometry(0.15, 16, 16);
        const ptsMat = new THREE.MeshStandardMaterial({ roughness: 0.28, metalness: 0.25 });
        const instPts = new THREE.InstancedMesh(sphereGeo, ptsMat, nPts);
        instPts.castShadow = !isSoftwareGl;

        const positions: THREE.Vector3[] = [];
        for (let i = 0; i < nPts; i++) {
          const cIdx = i % centroids.length;
          const c = centroids[cIdx].pos;
          const p = new THREE.Vector3(
            c.x + (rand() - 0.5) * 2.5,
            Math.max(0.28, c.y + (rand() - 0.5) * 1.9),
            c.z + (rand() - 0.5) * 2.5
          );
          positions.push(p);
          _dummyObj.position.copy(p);
          _dummyObj.scale.set(1, 1, 1);
          _dummyObj.updateMatrix();
          instPts.setMatrixAt(i, _dummyObj.matrix);
          _tmpColor.setHex(clusterColors[cIdx]);
          instPts.setColorAt(i, _tmpColor);
        }
        instPts.instanceMatrix.needsUpdate = true;
        if (instPts.instanceColor) instPts.instanceColor.needsUpdate = true;
        instPts.computeBoundingSphere();
        graphGroup.add(instPts);
        disposables.push(sphereGeo, ptsMat, instPts);

        centroids.forEach((c) => {
          addLabel(c.label, [c.pos.x, c.pos.y + 1.15, c.pos.z], c.color, 'rgba(255,255,255,0.95)', c.color, 2.35, 0.42);
        });

        const queryPos = new THREE.Vector3(0.2, 1.8, 0.1);
        const qGeo = new THREE.OctahedronGeometry(0.32, 0);
        const qMat = new THREE.MeshStandardMaterial({
          color: 0x312e81,
          roughness: 0.2,
          metalness: 0.6
        });
        const qMesh = new THREE.Mesh(qGeo, qMat);
        qMesh.position.copy(queryPos);
        graphGroup.add(qMesh);
        disposables.push(qGeo, qMat);

        addLabel('q (0.2, 1.8, 0.1) k=9', [queryPos.x, queryPos.y + 0.85, queryPos.z], '#1e1b4b', '#eef2ff', '#4f46e5', 2.45, 0.44);

        const k9Neighbors = [...positions]
          .sort((a, b) => a.distanceToSquared(queryPos) - b.distanceToSquared(queryPos))
          .slice(0, 9);

        const lineVerts: number[] = [];
        k9Neighbors.forEach((n) => {
          lineVerts.push(queryPos.x, queryPos.y, queryPos.z, n.x, n.y, n.z);
        });
        const lineGeo = new THREE.BufferGeometry();
        lineGeo.setAttribute('position', new THREE.Float32BufferAttribute(lineVerts, 3));
        const lineMat = new THREE.LineBasicMaterial({ color: 0x4f46e5 });
        graphGroup.add(new THREE.LineSegments(lineGeo, lineMat));
        disposables.push(lineGeo, lineMat);
      } else {
        buildCoordinateFrame(
          'X: Batch [0..5]',
          'Y: Layer [0..3]',
          'Z: Seq [0..5]',
          ['B=0', 'B=2', 'B=5'],
          ['L=0', 'L=1', 'L=3'],
          ['S=0', 'S=2', 'S=5']
        );

        const nx = 6;
        const ny = 4;
        const nz = 6;
        const total = nx * ny * nz;
        const voxGeo = new THREE.BoxGeometry(0.54, 0.54, 0.54);
        const voxMat = new THREE.MeshStandardMaterial({ roughness: 0.25, metalness: 0.22 });
        const instVox = new THREE.InstancedMesh(voxGeo, voxMat, total);
        instVox.castShadow = !isSoftwareGl;

        let idx = 0;
        for (let y = 0; y < ny; y++) {
          for (let x = 0; x < nx; x++) {
            for (let z = 0; z < nz; z++) {
              const act = Math.abs(Math.sin(x * 0.7 + y * 0.9 + z * 0.5 + rand() * 0.4));
              const s = 0.4 + act * 0.62;
              _dummyObj.position.set((x - 2.5) * 1.1, 0.48 + y * 0.92, (z - 2.5) * 1.1);
              _dummyObj.scale.set(s, s, s);
              _dummyObj.updateMatrix();
              instVox.setMatrixAt(idx, _dummyObj.matrix);
              _tmpColor.setHSL(0.62 - act * 0.35, 0.78, 0.42 + act * 0.16);
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

        addLabel('Tensor Max (5, 3, 5)', [2.75, 4.15, 2.75], '#312e81', '#eef2ff', '#6366f1', 2.25, 0.44);
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
        <div className="pointer-events-none absolute top-3 left-3 bg-white/95 border border-slate-300 rounded-xl px-3 py-1.5 shadow-xs max-w-sm">
          <div className="flex items-center gap-1.5 text-[10.5px] font-mono text-indigo-800 font-bold">
            <Box className="w-3.5 h-3.5" />
            <span>Interactive 3D WebGL Manifold (XYZ)</span>
          </div>
          <div className="text-xs font-bold text-slate-900 truncate mt-0.5">{project.title}</div>
          <div className="text-[10px] font-mono text-slate-600 truncate">
            {activeTechName} · {activeTypeName}
          </div>
        </div>

        {/* Top-Right Compact XYZ Axis Key */}
        <div className="pointer-events-none hidden sm:flex absolute top-3 right-3 bg-white/95 border border-slate-300 rounded-xl px-3 py-1.5 shadow-xs items-center gap-2 text-[10px] font-mono text-slate-700">
          <span className="font-bold text-rose-800">X</span>
          <span className="font-bold text-emerald-800">Y</span>
          <span className="font-bold text-blue-800">Z</span>
          <span className="text-slate-600">{axisSummary}</span>
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
            <div className="w-full bg-white/95 border border-slate-300 rounded-xl px-3 py-2 shadow-md flex flex-wrap items-center justify-between gap-2">
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
                        ? 'bg-indigo-600 text-white border-indigo-600'
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
          )}
        </div>
      </div>
    </div>
  );
}
