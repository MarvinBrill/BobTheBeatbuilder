import type { HitObject } from "../types";

function curvePosition(points: number[][], progress: number) {
  const samples = Array.from({ length: 65 }, (_, index) => {
    const t = index / 64;
    return [0, 1].map(
      (axis) =>
        (1 - t) ** 3 * points[0][axis] +
        3 * (1 - t) ** 2 * t * points[1][axis] +
        3 * (1 - t) * t * t * points[2][axis] +
        t ** 3 * points[3][axis],
    );
  });
  const lengths = [0];
  for (let i = 1; i < samples.length; i++)
    lengths.push(
      lengths[i - 1] +
        Math.hypot(
          samples[i][0] - samples[i - 1][0],
          samples[i][1] - samples[i - 1][1],
        ),
    );
  const target = progress * lengths.at(-1)!;
  const index = Math.max(
    1,
    lengths.findIndex((length) => length >= target),
  );
  const fraction =
    (target - lengths[index - 1]) /
    Math.max(0.001, lengths[index] - lengths[index - 1]);
  return samples[index - 1].map(
    (value, axis) => value + fraction * (samples[index][axis] - value),
  );
}

export function PreviewObjects({
  objects,
  time,
  cs,
  ar,
}: {
  objects: HitObject[];
  time: number;
  cs: number;
  ar: number;
}) {
  const radius = 54.4 - 4.48 * cs;
  const preempt = ar <= 5 ? 1800 - 120 * ar : 1200 - 150 * (ar - 5);
  let combo = 0;
  const visible = objects
    .map((object, index) => {
      combo = object.new_combo || index === 0 ? 1 : combo + 1;
      return { ...object, index, combo };
    })
    .filter(
      (object) =>
        object.time >= time - object.duration - 150 &&
        object.time <= time + preempt,
    )
    .slice(0, 50);

  return (
    <>
      {visible.reverse().map((object) => {
        const approach = 1 + Math.max(0, (object.time - time) / preempt) * 2;
        const curve = object.curve;
        const path =
          curve?.length === 4
            ? `M ${curve[0].join(" ")} C ${curve[1].join(" ")}, ${curve[2].join(" ")}, ${curve[3].join(" ")}`
            : `M ${object.x} ${object.y} L ${object.end_x} ${object.end_y}`;
        const traversals =
          Math.max(
            0,
            Math.min(
              0.999999,
              (time - object.time) / Math.max(1, object.duration),
            ),
          ) * (object.repeats || 1);
        const fraction =
          Math.floor(traversals) % 2 ? 1 - (traversals % 1) : traversals % 1;
        const following =
          object.slider &&
          time >= object.time &&
          time <= object.time + object.duration;
        const ball =
          following && curve?.length === 4
            ? curvePosition(curve, fraction)
            : null;
        return (
          <g
            key={object.index}
            opacity={object.time + object.duration < time ? 0.35 : 1}
          >
            {object.slider && (
              <>
                <path
                  d={path}
                  fill="none"
                  stroke="#b491db"
                  strokeWidth={radius * 2}
                  strokeLinecap="round"
                  opacity=".6"
                />
                <path
                  d={path}
                  fill="none"
                  stroke="#615477"
                  strokeWidth={Math.max(2, radius * 2 - 5)}
                  strokeLinecap="round"
                />
                {(object.repeats || 1) > 1 && (
                  <text
                    x={curve?.[3]?.[0] ?? object.end_x}
                    y={(curve?.[3]?.[1] ?? object.end_y) + 5}
                    fill="white"
                    textAnchor="middle"
                    fontSize={14}
                  >
                    ↶
                  </text>
                )}
              </>
            )}
            {time < object.time && (
              <circle
                cx={object.x}
                cy={object.y}
                r={radius * approach}
                fill="none"
                stroke="#f7a8d3"
                strokeWidth="1.5"
                opacity=".6"
              />
            )}
            <circle
              cx={object.x}
              cy={object.y}
              r={radius}
              fill="#715579"
              stroke="#ffbedf"
              strokeWidth="2"
            />
            <text
              x={object.x}
              y={object.y + 5}
              fill="white"
              fontSize={Math.min(18, radius)}
              textAnchor="middle"
            >
              {object.combo}
            </text>
            {ball && (
              <circle
                cx={ball[0]}
                cy={ball[1]}
                r={radius * 0.7}
                fill="#ffbedf"
                stroke="white"
                strokeWidth="2"
              />
            )}
          </g>
        );
      })}
    </>
  );
}
