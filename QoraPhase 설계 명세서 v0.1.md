# QoraPhase 설계 명세서 v0.1

## 1. 프로젝트 목표

**QoraPhase**는 JAX 기반의 현대적 CDI/BCDI phase retrieval 프레임워크다.

핵심 목표는 두 가지다.

- 일반 사용자는 reconstruction을 간단히 실행할 수 있어야 한다.
- 연구자는 projection, forward model, solver를 직접 조합하거나 교체할 수 있어야 한다.

핵심 철학:

> **Simple things are one function call; research algorithms are composable objects.**

---

# 2. 기본 범위

v0.1에서는 다음만 지원한다.

- 2D / 3D CDI
- complex-valued object
- ER
- HIO
- RAAR
- modulus projection
- support projection
- shrinkwrap
- random initialization
- multi-start reconstruction
- 기본 error metric
- JAX JIT / GPU acceleration

BCDI geometry, NUFFT, strain 계산, autodiff optimizer 등은 이후 버전으로 미룬다.

---

# 3. 입력 철학

QoraPhase는 **파일 포맷을 관리하지 않는다.**

기본 입력은 배열이다.

```python
import qoraphase as qp

result = qp.reconstruct(
    intensity,
    support=support,
    mask=mask,
)
```

다음 입력을 자연스럽게 받아야 한다.

```python
numpy.ndarray
jax.Array
```

public API boundary에서:

```python
jnp.asarray(...)
```

로 변환한다.

파일 읽기는 사용자가 담당한다.

```python
with h5py.File("scan.h5") as f:
    intensity = f["data"][:]

result = qp.reconstruct(intensity)
```

즉 구조는:

```text
User / Beamline I/O
        ↓
     ndarray
        ↓
    QoraPhase
        ↓
Reconstruction
```

---

# 4. NumPy와 JAX 역할

## NumPy

주로:

- 파일 입출력
- preprocessing
- plotting
- 외부 라이브러리 compatibility

에 사용한다.

## JAX

reconstruction core 전체에 사용한다.

- FFT
- projection
- ER/HIO/RAAR
- shrinkwrap
- batching
- JIT
- GPU
- autodiff 확장

core 내부에서는 `numpy` 연산을 섞지 않는다.

---

# 5. 사용자 API

## 가장 단순한 사용

```python
import qoraphase as qp

result = qp.reconstruct(
    intensity,
    support=support,
)

result.plot()
```

기본값으로 적절한 reconstruction recipe를 사용한다.

---

## 일반 연구자

```python
problem = qp.CDI(
    intensity=intensity,
    support=support,
    mask=mask,
)

recipe = qp.Recipe(
    qp.RAAR(800, beta=0.9),
    qp.ER(200),
)

result = problem.reconstruct(recipe)
```

---

## Multi-start

```python
ensemble = problem.reconstruct(
    recipe,
    starts=32,
)

best = ensemble.best
```

multi-start는 부가 기능이 아니라 기본 기능으로 취급한다.

---

# 6. 핵심 데이터 구조

## CDIProblem

고정된 reconstruction 입력을 보관한다.

```python
CDIProblem(
    intensity,
    support,
    mask,
    forward_model,
)
```

iteration 상태는 포함하지 않는다.

---

## ReconState

iteration 중 변하는 값만 가진다.

```python
class ReconState(NamedTuple):
    obj: jax.Array
    support: jax.Array
    iteration: jax.Array
```

operator는 state를 직접 수정하지 않는다.

```python
new_state = operator(state, problem)
```

---

# 7. Forward Model

FFT를 알고리즘 안에 직접 박지 않는다.

```python
class ForwardModel:
    def forward(self, obj):
        ...

    def adjoint(self, wave):
        ...
```

v0.1:

```python
qp.FFT()
```

향후:

```python
qp.NUFFT()
qp.NonOrthogonalFFT()
qp.Fresnel()
```

이 구조를 통해 HIO/RAAR는 그대로 두고 forward model만 교체할 수 있어야 한다.

---

# 8. Projection 구조

핵심 constraint는 독립된 projection으로 만든다.

```python
qp.ModulusProjection()
qp.SupportProjection()
```

ER, HIO, RAAR는 이 projection들을 이용한다.

예:

```python
qp.ER(200)

qp.HIO(
    500,
    beta=0.9,
)

qp.RAAR(
    800,
    beta=0.9,
)
```

첫 번째 positional argument는 항상 iteration 수다.

---

# 9. Recipe

여러 stage를 선언적으로 조합한다.

```python
recipe = qp.Recipe(
    qp.RAAR(800),
    qp.ER(200),
)
```

코드 순서가 실행 순서다.

PyNX처럼:

```python
ER()**200 * RAAR()**800
```

같은 operator-overloading 문법은 사용하지 않는다.

읽기 쉬운 명시적 API를 우선한다.

---

# 10. 반복과 Shrinkwrap

```python
recipe = qp.Recipe(
    qp.repeat(
        10,
        qp.RAAR(50),
        qp.Shrinkwrap(
            threshold=0.2,
            sigma=2.0,
        ),
    ),
    qp.ER(200),
)
```

향후 parameter schedule도 지원한다.

```python
qp.RAAR(
    800,
    beta=qp.LinearSchedule(
        0.95,
        0.8,
    ),
)
```

---

# 11. JAX 실행 모델

반복 계산에는:

```python
jax.lax.scan
```

multi-start에는:

```python
jax.vmap
```

전체 계산에는 가능한 경우:

```python
jax.jit
```

을 사용한다.

구조:

```text
Recipe
  ↓
Validation
  ↓
JAX execution graph
  ↓
JIT compile
  ↓
CPU / GPU
```

---

# 12. Reconstruction 결과

single reconstruction 결과:

```python
result.object
result.amplitude
result.phase
result.support
result.metrics
```

`amplitude`, `phase`는 `object`에서 계산되는 property로 둔다.

```python
result.amplitude
# abs(object)

result.phase
# angle(object)
```

---

# 13. Ensemble

여러 reconstruction은 `Ensemble`로 표현한다.

```python
ensemble.best
ensemble.runs
ensemble.ranking
```

Python collection처럼 동작해야 한다.

```python
len(ensemble)

ensemble[0]

ensemble[:10]

for result in ensemble:
    ...
```

향후:

```python
ensemble.average()
ensemble.align()
ensemble.cluster()
ensemble.prtf()
```

를 추가한다.

---

# 14. Randomness

JAX PRNG 방식을 따른다.

사용자는 간단히:

```python
result = qp.reconstruct(
    intensity,
    seed=42,
)
```

로 사용할 수 있다.

multi-start에서는 각 run의 seed가 재현 가능해야 한다.

---

# 15. Device 처리

기본적으로 사용자가 GPU 여부를 신경 쓰지 않게 한다.

```python
result = qp.reconstruct(...)
```

JAX의 기본 device selection을 따른다.

필요한 경우에만:

```python
problem.reconstruct(
    device="gpu:0",
)
```

처럼 지정한다.

---

# 16. 패키지 구조

초기에는 단순하게 유지한다.

```text
qoraphase/
├── __init__.py
├── api.py
│
├── core/
│   ├── problem.py
│   ├── state.py
│   ├── result.py
│   └── ensemble.py
│
├── operators/
│   ├── fourier.py
│   ├── modulus.py
│   ├── support.py
│   └── shrinkwrap.py
│
├── algorithms/
│   ├── er.py
│   ├── hio.py
│   └── raar.py
│
├── models/
│   └── fft.py
│
├── metrics/
│   └── errors.py
│
└── recipes/
    └── recipe.py
```

필요하기 전까지 I/O, plotting, CLI 등을 별도 대형 subsystem으로 만들지 않는다.

---

# 17. Public API

사용자는 기본적으로:

```python
import qoraphase as qp
```

만 사용한다.

주요 public symbol:

```python
qp.reconstruct

qp.CDI

qp.Recipe

qp.ER
qp.HIO
qp.RAAR

qp.Shrinkwrap

qp.FFT
```

내부 package path에 의존하는 사용법은 권장하지 않는다.

---

# 18. Type hints

현대 Python typing을 적극 사용한다.

```python
def reconstruct(
    intensity: ArrayLike,
    *,
    support: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    recipe: Recipe | None = None,
    starts: int = 1,
    seed: int | None = None,
) -> Reconstruction | Ensemble:
    ...
```

IDE autocomplete과 static analysis를 중요하게 본다.

---

# 19. v0.1 구현 목표

첫 버전에서 반드시 구현할 것:

```text
CDIProblem
ReconState

FFT

ModulusProjection
SupportProjection

ER
HIO
RAAR

RandomInit
Shrinkwrap

Recipe

FourierError

reconstruct()

Reconstruction
Ensemble

JIT
multi-start
```

파일 포맷 지원은 v0.1 필수 기능이 아니다.

---

# 20. 향후 확장

## v0.2

- reconstruction ranking
- alignment / averaging
- richer metrics
- callbacks
- checkpointing
- schedules

## v0.3

- BCDI geometry
- non-orthogonal transform
- reciprocal-space coordinate handling
- phase → displacement / strain

## v0.4

- autodiff losses
- Adam / LBFGS
- regularization
- projection + gradient hybrid reconstruction

---

# 21. 최종 설계 원칙

구현 판단의 우선순위는 다음과 같다.

1. 수학적 의미가 코드에서 명확할 것
2. 사용자 API가 짧고 읽기 쉬울 것
3. algorithm과 forward model이 분리될 것
4. JAX transform과 자연스럽게 동작할 것
5. mutation과 hidden global state를 최소화할 것
6. 재현성을 보장할 것
7. premature abstraction을 피할 것

QoraPhase의 핵심 정체성은 다음과 같다.

> **A JAX-native, composable phase-retrieval framework for coherent diffraction imaging.**

그리고 가장 중요한 UX 목표는 이것이다.

```python
import qoraphase as qp

result = qp.reconstruct(
    intensity,
    support=support,
)
```

**Array in → reconstruction out.**

파일을 어디서 읽었는지, 어떤 beamline에서 왔는지는 QoraPhase core의 관심사가 아니다.