"""End-to-end tests for the device-replacement license handover feature.

A handover atomically revokes one currently effective license and approves a
pending replacement on two consecutive revisions (revocation first).  The
replacement is adjudicated against the effective set *with the original
license excluded* under the live guard band, and against every active
restriction window.  A waitlisted replacement leaves the queue first; after
the two handover revisions the remaining FIFO waitlist is re-adjudicated by
the existing release rules.

A stale revision, a changed status, a remaining interference object or a
restriction-window hit aborts the whole operation: licenses, queue and
revisions stay untouched.  Guard band is the test default (2.0); every
application shares the default coverage circle and UTC window unless stated,
so only the frequency bands decide interference.
"""
from factories import make_payload


def submit(client, low, high, name="Event", **overrides):
    payload = make_payload(freq=(low, high), name=name, **overrides)
    r = client.post("/api/applications", json=payload)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def approve(client, app_id, expected):
    r = client.post(
        f"/api/applications/{app_id}/approve",
        json={"expected_revision": expected},
    )
    assert r.status_code == 200, r.text
    return r.json()


def revoke(client, app_id, reason=""):
    return client.post(
        f"/api/applications/{app_id}/revoke", json={"reason": reason}
    )


def register(client, app_id, reason=""):
    return client.post(
        f"/api/applications/{app_id}/waitlist", json={"reason": reason}
    )


def queue_ids(client):
    return client.get("/api/waitlist").json()["waitlist_application_ids"]


def current_revision(client):
    return client.get("/api/revision/current").json()["current_revision"]


def preview_handover(client, original, replacement, expected, reason="swap"):
    return client.post(
        "/api/license-handovers/preview",
        json={
            "original_application_id": original,
            "replacement_application_id": replacement,
            "expected_revision": expected,
            "reason": reason,
        },
    )


def confirm_handover(client, original, replacement, expected, reason="swap"):
    return client.post(
        "/api/license-handovers",
        json={
            "original_application_id": original,
            "replacement_application_id": replacement,
            "expected_revision": expected,
            "reason": reason,
        },
    )


class TestPreview:
    def test_preview_excludes_original_and_reports_effective_set(self, client):
        a = submit(client, 100, 110, "A")
        approve(client, a, 0)  # rev 1
        b = submit(client, 105, 108, "B")  # overlaps A, replaces it
        # Another effective license E disjoint from B.
        e = submit(client, 200, 210, "E")
        approve(client, e, 1)  # rev 2

        r = preview_handover(client, a, b, expected=2)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["current_revision"] == 2
        assert body["revision_matches"] is True
        assert body["can_proceed"] is True
        # The original is excluded; E stays in the judged effective set.
        assert body["excluded_application_id"] == a
        assert body["effective_application_ids"] == [e]
        assert body["conflicts"] == []
        assert body["has_conflicts"] is False
        assert body["restriction_window_conflicts"] == []
        assert body["replacement_waitlisted"] is False
        assert body["replacement_queue_position"] is None
        # Read-only: nothing moved.
        assert current_revision(client) == 2
        assert client.get(f"/api/applications/{a}").json()["status"] == "approved"

    def test_preview_is_read_only_when_blocked(self, client):
        a = submit(client, 100, 110, "A")
        e = submit(client, 200, 205, "E")
        b = submit(client, 201, 206, "B")  # clashes with E once A is gone
        approve(client, a, 0)
        approve(client, e, 1)

        r = preview_handover(client, a, b, expected=2)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["can_proceed"] is False
        assert body["has_conflicts"] is True
        assert [c["application"]["id"] for c in body["conflicts"]] == [e]
        # State is untouched.
        assert current_revision(client) == 2
        assert client.get(f"/api/applications/{b}").json()["status"] == "pending"

    def test_preview_reports_stale_expected_revision(self, client):
        a = submit(client, 100, 110, "A")
        approve(client, a, 0)
        b = submit(client, 200, 210, "B")
        body = preview_handover(client, a, b, expected=0).json()
        assert body["current_revision"] == 1
        assert body["revision_matches"] is False
        assert body["can_proceed"] is False
        # No revision consumed by the rehearsal.
        assert current_revision(client) == 1

    def test_preview_reports_waitlisted_replacement_position(self, client):
        a = submit(client, 100, 110, "A")
        approve(client, a, 0)
        b = submit(client, 200, 210, "B")
        register(client, b)
        body = preview_handover(client, a, b, expected=1).json()
        assert body["replacement_waitlisted"] is True
        assert body["replacement_queue_position"] == 1

    def test_preview_rejects_wrong_states(self, client):
        a = submit(client, 100, 110, "A")
        approve(client, a, 0)  # rev 1
        b = submit(client, 200, 210, "B")
        approve(client, b, 1)  # rev 2; B is approved, not pending
        c = submit(client, 300, 310, "C")

        # Original revoked -> not effective.
        revoke(client, a)  # rev 3
        r = preview_handover(client, a, c, expected=3)
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "not_effective"
        # Replacement no longer pending.
        r = preview_handover(client, b, a, expected=3)
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "replacement_not_pending"
        assert r.json()["error"]["details"]["status"] == "revoked"

    def test_preview_unknown_and_self_rejected(self, client):
        a = submit(client, 100, 110, "A")
        approve(client, a, 0)
        b = submit(client, 200, 210, "B")
        r = preview_handover(client, 999999, b, expected=1)
        assert r.status_code == 404
        assert r.json()["error"]["details"]["missing_application_ids"] == [
            999999
        ]
        r = preview_handover(client, a, 999999, expected=1)
        assert r.status_code == 404
        r = preview_handover(client, a, a, expected=1)
        assert r.status_code == 422
        assert r.json()["error"]["details"][0]["code"] == "same_application"


class TestConfirmSuccess:
    def test_handover_revokes_and_approves_on_consecutive_revisions(self, client):
        a = submit(client, 100, 110, "A")
        approve(client, a, 0)  # rev 1
        b = submit(client, 105, 108, "B")  # overlaps A, takes its place

        r = confirm_handover(client, a, b, expected=1)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["handed_over"] is True
        assert body["revocation_revision"] == 2
        assert body["approval_revision"] == 3
        assert body["final_revision"] == 3
        assert body["revision"] == 2
        assert body["original"]["status"] == "revoked"
        assert body["replacement"]["status"] == "approved"
        assert body["approval"]["revision"] == 3
        assert body["approval"]["handover_approved"] is True
        assert body["approval"]["handover_revocation_revision"] == 2
        # No drain participants: lists are empty.
        assert body["auto_approved_application_ids"] == []
        assert body["remaining_waitlist_application_ids"] == []

        assert client.get(f"/api/applications/{a}").json()["status"] == "revoked"
        repl = client.get(f"/api/applications/{b}").json()
        assert repl["status"] == "approved"
        assert repl["approval_revision"] == 3
        assert current_revision(client) == 3

    def test_original_and_replacement_need_not_be_compatible(self, client):
        # The replacement sits exactly inside the original's band: without the
        # exclusion this could never be approved; with it, it is free.
        a = submit(client, 100, 110, "A")
        approve(client, a, 0)  # rev 1
        b = submit(client, 102, 108, "B")
        r = confirm_handover(client, a, b, expected=1)
        assert r.status_code == 200, r.text
        assert r.json()["approval_revision"] == 3

    def test_waitlisted_replacement_leaves_queue_before_drain(self, client):
        a = submit(client, 100, 110, "A")
        approve(client, a, 0)  # rev 1
        b = submit(client, 105, 108, "B")  # replacement, queued
        d = submit(client, 200, 210, "D")  # clean stand-by
        e = submit(client, 202, 209, "E")  # clashes with D, stays queued
        register(client, b)
        register(client, d)
        register(client, e)
        assert queue_ids(client) == [b, d, e]

        body = confirm_handover(client, a, b, expected=1).json()
        # Handover revoke = 2, replacement approval = 3, drain: D -> 4, E kept.
        assert body["revocation_revision"] == 2
        assert body["approval_revision"] == 3
        assert body["final_revision"] == 4
        assert body["auto_approved_application_ids"] == [d]
        assert body["remaining_waitlist_application_ids"] == [e]
        pending = body["waitlist_pending"]
        assert [p["application_id"] for p in pending] == [e]
        assert [c["application"]["id"] for c in pending[0]["conflicts"]] == [d]
        # The replacement is no longer queued and was never auto-approved.
        assert queue_ids(client) == [e]
        at4 = client.get("/api/revision/4").json()
        b_record = [x for x in at4["approvals"] if x["application_id"] == b][0]
        assert b_record["auto_approved"] is False
        assert b_record["release_revision"] is None
        d_record = [x for x in at4["approvals"] if x["application_id"] == d][0]
        assert d_record["auto_approved"] is True
        # The drain's release revision is the handover revocation revision.
        assert d_record["release_revision"] == 2

    def test_drain_uses_live_guard_band(self, client):
        # Tighten the band to 1 first; a stand-by 1.5 MHz away from the
        # replacement passes under the live band during the drain.
        a = submit(client, 100, 110, "A")
        approve(client, a, 0)  # rev 1
        r = client.post(
            "/api/config/guard-band",
            json={"guard_band": 1.0, "expected_revision": 1},
        )
        assert r.status_code == 200, r.text  # rev 2
        b = submit(client, 105, 108, "B")
        # C is 1.5 MHz from B's upper edge? Place C at 109.5-109.8: gap to
        # B's 108 is 1.5 > 1.0 -> free once A is gone.
        c = submit(client, 109.5, 109.8, "C")
        register(client, c)
        body = confirm_handover(client, a, b, expected=2).json()
        assert body["guard_band"] == 1.0
        assert body["auto_approved_application_ids"] == [c]
        assert body["auto_approvals"][0]["guard_band"] == 1.0
        assert body["final_revision"] == 5  # revoke 3, approve 4, drain 5


class TestConfirmRejections:
    def test_stale_revision_changes_nothing(self, client):
        a = submit(client, 100, 110, "A")
        approve(client, a, 0)
        b = submit(client, 200, 210, "B")
        r = confirm_handover(client, a, b, expected=0)
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "stale_revision"
        details = r.json()["error"]["details"]
        assert details == {"expected_revision": 0, "current_revision": 1}
        assert current_revision(client) == 1
        assert client.get(f"/api/applications/{a}").json()["status"] == "approved"
        assert client.get(f"/api/applications/{b}").json()["status"] == "pending"

    def test_original_state_changed_rejected(self, client):
        a = submit(client, 100, 110, "A")
        approve(client, a, 0)  # rev 1
        b = submit(client, 200, 210, "B")
        # Revoke A independently first (rev 2).
        revoke(client, a)
        r = confirm_handover(client, a, b, expected=2)
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "not_effective"
        # Revision unchanged; B stays pending.
        assert current_revision(client) == 2
        assert client.get(f"/api/applications/{b}").json()["status"] == "pending"

    def test_suspended_original_rejected(self, client):
        a = submit(client, 100, 110, "A")
        approve(client, a, 0)
        b = submit(client, 200, 210, "B")
        client.post(
            f"/api/applications/{a}/suspend",
            json={"expected_revision": 1, "reason": "outage"},
        )
        r = confirm_handover(client, a, b, expected=2)
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "not_effective"
        # A stays suspended (not revoked), revision unchanged.
        assert client.get(f"/api/applications/{a}").json()["status"] == "suspended"
        assert current_revision(client) == 2

    def test_replacement_state_changed_rejected(self, client):
        a = submit(client, 100, 110, "A")
        approve(client, a, 0)  # rev 1
        b = submit(client, 200, 210, "B")
        approve(client, b, 1)  # rev 2
        r = confirm_handover(client, a, b, expected=2)
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "replacement_not_pending"
        # A remains effective, revision unchanged.
        assert client.get(f"/api/applications/{a}").json()["status"] == "approved"
        assert current_revision(client) == 2

    def test_interference_with_other_effective_license_rejected(self, client):
        a = submit(client, 100, 110, "A")
        e = submit(client, 200, 205, "E")
        b = submit(client, 201, 206, "B")  # clashes with E once A is excluded
        approve(client, a, 0)
        approve(client, e, 1)
        r = confirm_handover(client, a, b, expected=2)
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "interference"
        conflicts = r.json()["error"]["details"]["conflicts"]
        assert [c["application"]["id"] for c in conflicts] == [e]
        assert current_revision(client) == 2
        assert client.get(f"/api/applications/{a}").json()["status"] == "approved"
        assert client.get(f"/api/applications/{b}").json()["status"] == "pending"
        assert queue_ids(client) == []

    def test_restriction_window_hit_rejected(self, client):
        # Future-day scene so the window is not expired.
        window = "2030-10-01T08:00:00Z", "2030-10-01T12:00:00Z"
        a = submit(client, 100, 110, "A", start=window[0], end=window[1])
        approve(client, a, 0)  # rev 1
        b = submit(client, 200, 205, "B", start=window[0], end=window[1])
        # Window over B's own band; A is untouched.
        r = client.post(
            "/api/restriction-windows",
            json={
                "circle": {"cx": 0, "cy": 0, "radius": 1000},
                "frequency": {"low": 200, "high": 205},
                "time_window": {"start": window[0], "end": window[1]},
                "reason": "control",
                "expected_revision": 1,
            },
        )
        assert r.status_code == 201, r.text  # rev 2
        r = confirm_handover(client, a, b, expected=2)
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "restriction_window"
        hits = r.json()["error"]["details"][
            "restriction_window_conflicts"
        ]
        assert len(hits) == 1
        assert current_revision(client) == 2
        assert client.get(f"/api/applications/{a}").json()["status"] == "approved"
        assert client.get(f"/api/applications/{b}").json()["status"] == "pending"

    def test_unknown_ids_and_self_handover_rejected(self, client):
        a = submit(client, 100, 110, "A")
        approve(client, a, 0)
        b = submit(client, 200, 210, "B")
        r = confirm_handover(client, 999999, b, expected=1)
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "not_found"
        r = confirm_handover(client, a, a, expected=1)
        assert r.status_code == 422
        assert r.json()["error"]["details"][0]["code"] == "same_application"
        assert current_revision(client) == 1

    def test_aborted_handover_keeps_waitlist_intact(self, client):
        a = submit(client, 100, 110, "A")
        e = submit(client, 200, 205, "E")
        b = submit(client, 201, 206, "B")  # clashes with E
        d = submit(client, 300, 310, "D")
        approve(client, a, 0)
        approve(client, e, 1)
        register(client, b)
        register(client, d)
        r = confirm_handover(client, a, b, expected=2)
        assert r.status_code == 409
        # Everything is untouched: original effective, B pending and queued,
        # D queued, revision unchanged.
        assert current_revision(client) == 2
        assert client.get(f"/api/applications/{a}").json()["status"] == "approved"
        assert queue_ids(client) == [b, d]


class TestHistory:
    def test_snapshots_restore_each_handover_revision(self, client):
        a = submit(client, 100, 110, "A")
        approve(client, a, 0)  # rev 1
        b = submit(client, 105, 108, "B")
        confirm_handover(client, a, b, expected=1)  # revoke 2, approve 3

        at1 = client.get("/api/revision/1").json()
        assert at1["handovers"] == []

        at2 = client.get("/api/revision/2").json()
        assert at2["effective_application_ids"] == []
        assert at2["handovers"][0]["revocation_revision"] == 2
        # At the revocation revision the approval half is not visible yet.
        assert at2["handovers"][0]["approval_revision"] is None
        assert at2["handovers"][0]["complete"] is False
        revocation = [r for r in at2["revocations"] if r["application_id"] == a][0]
        assert revocation["handover_replacement_application_id"] == b
        assert revocation["handover_approval_revision"] == 3

        at3 = client.get("/api/revision/3").json()
        assert at3["effective_application_ids"] == [b]
        handover = at3["handovers"][0]
        assert handover["approval_revision"] == 3
        assert handover["final_revision"] == 3
        assert handover["complete"] is True
        approval = [x for x in at3["approvals"] if x["application_id"] == b][0]
        assert approval["handover_approved"] is True
        assert approval["handover_original_application_id"] == a
        assert approval["handover_revocation_revision"] == 2
        assert approval["auto_approved"] is False
        # A's original approval shows revoked as-of, and the replacement is
        # effective.
        a_record = [x for x in at3["approvals"] if x["application_id"] == a][0]
        assert a_record["status_as_of"] == "revoked"
        assert a_record["revoked_at_revision"] == 2

    def test_current_snapshot_lists_handovers(self, client):
        a = submit(client, 100, 110, "A")
        approve(client, a, 0)
        b = submit(client, 200, 210, "B")
        confirm_handover(client, a, b, expected=1, reason="device swap")
        cur = client.get("/api/revision/current").json()
        assert len(cur["handovers"]) == 1
        h = cur["handovers"][0]
        assert h["original_application_id"] == a
        assert h["replacement_application_id"] == b
        assert h["reason"] == "device swap"
        assert (h["revocation_revision"], h["approval_revision"]) == (2, 3)

    def test_comparison_restores_handover_and_status_changes(self, client):
        a = submit(client, 100, 110, "A")
        approve(client, a, 0)  # rev 1
        b = submit(client, 200, 210, "B")
        confirm_handover(client, a, b, expected=1)  # rev 2, 3

        body = client.get(
            "/api/revision/compare?base_revision=1&target_revision=3"
        ).json()
        assert body["effective_application_ids"]["removed"] == [a]
        assert body["effective_application_ids"]["added"] == [b]
        assert [(h["original_application_id"], h["replacement_application_id"])
                for h in body["handovers"]] == [(a, b)]
        # The unified difference stream carries the handover too.
        assert "license_handover" in [e["type"] for e in body["differences"]]
        changes = {
            (c["application_id"], c["from_status"], c["to_status"])
            for c in body["status_changes"]
        }
        assert (a, "approved", "revoked") in changes
        assert (b, "pending", "approved") in changes

    def test_two_handovers_chain_through_history(self, client):
        a = submit(client, 100, 110, "A")
        b = submit(client, 105, 108, "B")
        c = submit(client, 102, 106, "C")
        approve(client, a, 0)  # rev 1
        confirm_handover(client, a, b, expected=1)  # rev 2, 3
        confirm_handover(client, b, c, expected=3)  # rev 4, 5

        cur = client.get("/api/revision/current").json()
        assert cur["effective_application_ids"] == [c]
        assert [h["replacement_application_id"] for h in cur["handovers"]] == [
            b,
            c,
        ]
        statuses = {
            app_id: client.get(f"/api/applications/{app_id}").json()["status"]
            for app_id in (a, b, c)
        }
        assert statuses == {a: "revoked", b: "revoked", c: "approved"}
        # The intermediate state (B effective) is still restorable.
        at3 = client.get("/api/revision/3").json()
        assert at3["effective_application_ids"] == [b]
        at4 = client.get("/api/revision/4").json()
        assert at4["effective_application_ids"] == []


class TestExistingSemanticsPreserved:
    def test_plain_revoke_and_approve_have_no_handover_fields(self, client):
        a = submit(client, 100, 110, "A")
        approve(client, a, 0)  # rev 1
        revoke(client, a, reason="ordinary")  # rev 2
        cur = client.get("/api/revision/current").json()
        assert cur["handovers"] == []
        revocation = cur["revocations"][0]
        assert "handover_id" not in revocation
        approval = cur["approvals"][0]
        assert "handover_id" not in approval
        assert approval["auto_approved"] is False

    def test_waitlist_release_drain_still_works(self, client):
        a = submit(client, 100, 110, "A")
        approve(client, a, 0)
        b = submit(client, 200, 210, "B")
        register(client, b)
        body = revoke(client, a).json()
        assert body["auto_approved_application_ids"] == [b]
        assert body["final_revision"] == 3
